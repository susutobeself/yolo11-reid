import time
import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from numpy import random
from ultralytics import YOLO
from utils.datasets import LoadStreams, LoadImages
from utils.plots import plot_one_box
from utils.extracter import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from common.baseclass import BaseClass

class PersonSearch(BaseClass):
    def __init__(self,
                 weights='models/yolo11m.pt',
                 reid_config='models/opts.yaml',
                 query_folder='files/queries',
                 device='0' if torch.cuda.is_available() else 'cpu',
                 img_size=640,
                 conf_thres=0.25,
                 match_threshold=0.7):
        
        super().__init__()

        self.device = device
        self.conf_thres = conf_thres
        self.match_threshold = match_threshold

        # 加载模型
        self.model = YOLO(weights)
        self.names = self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.img_size = img_size

        if 'person' not in self.names.values():
            raise ValueError("YOLO 模型不支持 'person' 类别，请使用 COCO 预训练模型")

        # 初始化 ReID 特征提取器
        self.extractor = FeatureExtractor(reid_config)
        self.target_features = {}  # {person_id: feature_vector}

        # 加载查询图片
        self.load_query_images(query_folder)

    def normalize_features(self, features):
        norm = np.linalg.norm(features)
        return features / norm if norm > 0 else features

    def cosine_similarity(self, feat1, feat2):
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

    def load_query_images(self, query_folder):
        if not os.path.exists(query_folder):
            raise FileNotFoundError(f"Query folder {query_folder} not found!")

        image_paths = glob.glob(os.path.join(query_folder, "*.jpg"))
        if not image_paths:
            raise FileNotFoundError(f"No jpg images found in {query_folder}")

        for img_path in sorted(image_paths):
            try:
                person_id = int(os.path.splitext(os.path.basename(img_path))[0])
                img = Image.open(img_path).convert("RGB")
                features = self.extractor.extract_feature(img)
                features = self.normalize_features(features)
                self.target_features[person_id] = features
                self.logdebug(f"load_query_images Loaded person ID: {person_id:04d}")
            except Exception as e:
                self.logerror(f"load_query_images Failed to process {img_path}: {e}")
        self.loginfo(f"load_query_images Loading query images from {query_folder},Successfully loaded {len(self.target_features)} query persons.\n")
    
    def time_synchronized(self):
        # pytorch-accurate time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    def process_frame(self, im0):
        t1 = self.time_synchronized()
        results = self.model(im0, imgsz=self.img_size, verbose=False)[0]

        boxes, crops = [], []
        box_coords = []

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if self.names[cls_id] != 'person' or conf < self.conf_thres:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w = im0.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                person_img = im0[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue

                person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)).resize((128, 256))
                crops.append(person_pil)
                box_coords.append((x1, y1, x2, y2))
                boxes.append(box)

        if not crops:
            return im0, []

        features = [self.normalize_features(self.extractor.extract_feature(img)) for img in crops]

        query_ids = list(self.target_features.keys())
        query_feats = np.stack([self.target_features[qid] for qid in query_ids])

        sim_matrix = cosine_similarity(features, query_feats)
        cost_matrix = 1 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_query = set()

        for det_idx, query_idx in zip(row_ind, col_ind):
            sim = sim_matrix[det_idx][query_idx]
            x1, y1, x2, y2 = box_coords[det_idx]
            if sim >= self.match_threshold:
                person_id = query_ids[query_idx]
                label = f'ID:{person_id:04d} ({sim:.2f})'
                plot_one_box([x1, y1, x2, y2], im0, label=label, color=[0, 0, 255], line_thickness=3)
                assigned_query.add(person_id)
            else:
                plot_one_box([x1, y1, x2, y2], im0, label='Unknown', color=[0, 255, 0], line_thickness=2)

        # 帧上绘制时间戳
        frame_time = int(time.time() * 1000)
        cv2.putText(im0, f"Time: {frame_time}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        self.logdebug(f'process_frame Done ({self.time_synchronized() - t1:.3f}s)')
        return im0, list(assigned_query)


    def search_video(self, source, view_img=True, save_path=None):
        if not self.target_features:
            raise ValueError("No query features loaded!")

        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        t0 = time.time()

        if save_path is None or os.path.isdir(save_path):
            if source.isnumeric():
                save_path = os.path.join(save_path or output_dir, f'camera_{source}_out.mp4')
            else:
                source_name = os.path.splitext(os.path.basename(source))[0]
                save_path = os.path.join(save_path or output_dir, f'{source_name}_out.mp4')


        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        dataset = LoadStreams(source) if webcam else LoadImages(source)

        vid_writer = None
        fps, w, h = 30, None, None

        for path, img, im0s, vid_cap in dataset:
            frames = im0s if webcam else [im0s]
            for im0 in frames:
                processed, matched_ids = self.process_frame(im0)

                # 保存匹配成功帧到每个 ID 的目录
                if matched_ids:
                    debug_dir = 'debug_matched_frames'
                    os.makedirs(debug_dir, exist_ok=True)
                    frame_time = int(time.time() * 1000)

                    for pid in matched_ids:
                        pid_dir = os.path.join(debug_dir, f'ID{pid:04d}')
                        os.makedirs(pid_dir, exist_ok=True)
                        save_path_img = os.path.join(pid_dir, f"frame_{frame_time}.jpg")
                        cv2.imwrite(save_path_img, processed)

                # 初始化写入器
                if save_path and vid_writer is None:
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        w, h = processed.shape[1], processed.shape[0]
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                if view_img:
                    cv2.imshow('ReID', processed)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return

                if vid_writer:
                    vid_writer.write(processed)

        if vid_writer:
            vid_writer.release()
            print(f"Results saved to {save_path}")
        print(f'Done. ({time.time() - t0:.3f}s)')
        cv2.destroyAllWindows()


