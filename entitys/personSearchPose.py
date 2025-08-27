import argparse  
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

class PersonSearchPose(BaseClass):
    def __init__(self,
                 weights='models/yolov8n-pose.pt',
                 reid_config='models/opts.yaml',
                 query_folder='files/queries',
                 device='0' if torch.cuda.is_available() else 'cpu',
                 img_size=640,
                 conf_thres=0.25,
                 match_threshold=0.7,
                 reid_weight=0.6,
                 pose_weight=0.3,
                 color_weight=0.1):
        
        super().__init__()

        self.device = device
        self.conf_thres = conf_thres
        self.match_threshold = match_threshold
        self.reid_weight = reid_weight
        self.pose_weight = pose_weight
        self.color_weight = color_weight

        # YOLOv8 姿态模型
        self.model = YOLO(weights)
        self.names = self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.img_size = img_size

        if 'person' not in self.names.values():
            raise ValueError("YOLO 模型不支持 'person' 类别，请使用 COCO 预训练模型")

        # ReID 特征提取器
        self.extractor = FeatureExtractor(reid_config)
        self.target_features = {}  # {person_id: {'reid': [orig, flip], 'pose': [orig, mirrored], 'color': feature}}
        self.best_matches = {}     # {person_id: {'score': float, 'img': np.array}}

        # 加载查询图片
        self.load_query_images(query_folder)

    @staticmethod
    def normalize_features(features, dim=None):
        if features is None or len(features) == 0:
            return np.zeros(dim, dtype=np.float32) if dim else np.array([])
        norm = np.linalg.norm(features)
        return features / norm if norm > 0 else features

    @staticmethod
    def extract_pose_features(keypoints, box=None):
        """提取骨架特征（相对坐标 + 角度 + 人体比例）"""
        if keypoints is None or len(keypoints) == 0:
            return np.zeros(14, dtype=np.float32)  # 默认14维

        kp_coords = keypoints.xy[0].cpu().numpy()[:, :2]
        kp_conf = keypoints.conf[0].cpu().numpy()

        # 缺失点插值
        mean_xy = np.nanmean(kp_coords, axis=0)
        kp_coords = np.where(np.isnan(kp_coords), mean_xy, kp_coords)

        # 关键点归一化
        if box is not None:
            x1, y1, x2, y2 = box
            box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
            kp_coords[:, 0] = (kp_coords[:, 0] - x1) / box_w
            kp_coords[:, 1] = (kp_coords[:, 1] - y1) / box_h

        # 稳定关键点：头顶、颈、左右肩、左右髋
        stable_idx = [0, 1, 2, 3, 5, 6]
        kp_coords = kp_coords[stable_idx]
        kp_conf = kp_conf[stable_idx]

        neck = kp_coords[1]
        rel_coords = kp_coords - neck

        # 夹角
        def angle(p1, p2):
            v = p2 - p1
            return np.arctan2(v[1], v[0])

        left_shoulder_angle = angle(kp_coords[2], kp_coords[5])
        left_hip_angle = angle(kp_coords[4], kp_coords[5])

        # 身体比例
        shoulder_width = np.linalg.norm(kp_coords[2] - kp_coords[3])
        hip_width = np.linalg.norm(kp_coords[4] - kp_coords[5])
        body_height = np.max(kp_coords[:, 1]) - np.min(kp_coords[:, 1])
        ratios = np.array([shoulder_width/body_height, hip_width/body_height], dtype=np.float32) if body_height>1e-6 else np.array([0.0,0.0],dtype=np.float32)

        # 特征向量
        vectors = np.concatenate([rel_coords.flatten(), [left_shoulder_angle, left_hip_angle], ratios])
        vectors = np.nan_to_num(vectors, nan=0.0)
        coord_norm = np.linalg.norm(vectors[:-4])
        if coord_norm > 0:
            vectors[:-4] /= coord_norm

        return vectors.astype(np.float32), kp_conf

    @staticmethod
    def mirror_pose_feature(kp_coords):
        """左右镜像"""
        mirror_idx = [0,1,3,2,5,4]
        return kp_coords[mirror_idx]

    @staticmethod
    def extract_color_hist(img):
        hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256]*3)
        hist = hist.flatten() / np.sum(hist)
        return hist

    def extract_part_based_feature(self, img: Image.Image):
        """Part-based ReID 特征，上/中/下三部分 + 全局融合"""
        img = img.resize((128, 256))
        w, h = img.size
        upper = img.crop((0, 0, w, h//3))
        middle = img.crop((0, h//3, w, 2*h//3))
        lower = img.crop((0, 2*h//3, w, h))

        feat_upper = self.normalize_features(self.extractor.extract_feature(upper))
        feat_middle = self.normalize_features(self.extractor.extract_feature(middle))
        feat_lower = self.normalize_features(self.extractor.extract_feature(lower))

        # 局部特征最大值融合
        feat_part = np.maximum.reduce([feat_upper, feat_middle, feat_lower])
        # 全局特征
        feat_global = self.normalize_features(self.extractor.extract_feature(img))
        # 融合全局 + 局部
        feat_final = self.normalize_features(0.5*feat_global + 0.5*feat_part)
        return feat_final

    def time_synchronized(self):
        # pytorch-accurate time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

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
                img_np = np.array(img)

                # ReID 特征（part-based + flip）
                reid_feat_orig = self.extract_part_based_feature(img)
                reid_feat_flip = self.extract_part_based_feature(img.transpose(Image.FLIP_LEFT_RIGHT))

                # Pose 特征
                results = self.model(img_np, verbose=False)
                pose_feat_orig, pose_conf = np.zeros(14), None
                pose_feat_mirror = np.zeros(14)
                if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
                    pose_feat_orig, pose_conf = self.extract_pose_features(results[0].keypoints[0])
                    pose_feat_mirror = self.mirror_pose_feature(pose_feat_orig)

                # 颜色直方图
                color_feat = self.extract_color_hist(img_np)

                self.target_features[person_id] = {
                    'reid': [reid_feat_orig, reid_feat_flip],
                    'pose': [pose_feat_orig, pose_feat_mirror],
                    'color': color_feat
                }
                self.logdebug(f"Loaded person ID: {person_id:04d} - ReID dim: {len(reid_feat_orig)}, Pose dim: {len(pose_feat_orig)}, Color dim: {len(color_feat)}")
            except Exception as e:
                self.logerror(f"Failed to process {img_path}: {e}")
        self.loginfo(f"Successfully loaded {len(self.target_features)} query persons.\n")

    def process_frame(self, im0):
        t1 = self.time_synchronized()
        results = self.model(im0, imgsz=self.img_size, verbose=False)[0]
        boxes, crops, box_coords, keypoints_list = [], [], [], []

        for box, kps in zip(results.boxes or [], results.keypoints or [None]*len(results.boxes)):
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
            person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
            crops.append(person_pil)
            box_coords.append((x1, y1, x2, y2))
            boxes.append(box)
            keypoints_list.append(kps)

        if not crops:
            return im0, []

        # ReID 特征（part-based）
        reid_features = []
        for img in crops:
            orig = self.extract_part_based_feature(img)
            flip = self.extract_part_based_feature(img.transpose(Image.FLIP_LEFT_RIGHT))
            reid_features.append([orig, flip])

        # Pose 特征
        pose_features = []
        pose_confs = []
        for i, kps in enumerate(keypoints_list):
            if kps is not None:
                feat, conf = self.extract_pose_features(kps, box_coords[i])
                pose_features.append([feat, self.mirror_pose_feature(feat)])
                pose_confs.append(conf)
            else:
                pose_features.append([np.zeros(14), np.zeros(14)])
                pose_confs.append(None)

        # 颜色特征
        color_features = [self.extract_color_hist(im0[y1:y2, x1:x2]) for x1, y1, x2, y2 in box_coords]

        query_ids = list(self.target_features.keys())
        combined_sim_matrix = np.zeros((len(crops), len(query_ids)))

        for q_idx, pid in enumerate(query_ids):
            q_reid_orig, q_reid_flip = self.target_features[pid]['reid']
            q_pose_orig, q_pose_mirror = self.target_features[pid]['pose']
            q_color = self.target_features[pid]['color']

            for d_idx in range(len(crops)):
                det_reid_orig, det_reid_flip = reid_features[d_idx]
                det_pose_orig, det_pose_mirror = pose_features[d_idx]
                det_color = color_features[d_idx]

                # ReID 双向取最大
                reid_sim = max(cosine_similarity([det_reid_orig], [q_reid_orig])[0][0],
                               cosine_similarity([det_reid_flip], [q_reid_flip])[0][0])
                # Pose 双向取最大
                pose_sim = max(cosine_similarity([det_pose_orig], [q_pose_orig])[0][0],
                               cosine_similarity([det_pose_mirror], [q_pose_mirror])[0][0])
                # 颜色相似度
                color_sim = cosine_similarity([det_color], [q_color])[0][0]

                combined_sim = (self.reid_weight*reid_sim +
                                self.pose_weight*pose_sim +
                                self.color_weight*color_sim)
                combined_sim_matrix[d_idx, q_idx] = combined_sim

        # 匈牙利匹配
        cost_matrix = 1 - combined_sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_query = set()

        for det_idx, query_idx in zip(row_ind, col_ind):
            combined_sim = combined_sim_matrix[det_idx, query_idx]
            if combined_sim < self.match_threshold:
                continue
            pid = query_ids[query_idx]
            x1, y1, x2, y2 = box_coords[det_idx]
            label = f'ID:{pid:04d} S:{combined_sim:.2f}'
            plot_one_box([x1, y1, x2, y2], im0, label=label, color=[0,0,255], line_thickness=3)
            assigned_query.add(pid)
            if pid not in self.best_matches or combined_sim > self.best_matches[pid]['score']:
                self.best_matches[pid] = {'score': combined_sim, 'img': im0.copy()}

        self.logdebug(f'Done. ({self.time_synchronized() - t1:.3f}s)')
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

                if matched_ids:
                    debug_dir = 'debug_matched_frames'
                    os.makedirs(debug_dir, exist_ok=True)
                    frame_time = int(time.time() * 1000)
                    for pid in matched_ids:
                        pid_dir = os.path.join(debug_dir, f'ID{pid:04d}')
                        os.makedirs(pid_dir, exist_ok=True)
                        save_path_img = os.path.join(pid_dir, f"frame_{frame_time}.jpg")
                        cv2.imwrite(save_path_img, processed)

                if save_path and vid_writer is None:
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        w, h = processed.shape[1], processed.shape[0]
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                if view_img:
                    cv2.imshow('ReID YOLOv8', processed)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return

                if vid_writer:
                    vid_writer.write(processed)

        best_match_dir = 'best_matches'
        os.makedirs(best_match_dir, exist_ok=True)
        for pid, data in self.best_matches.items():
            save_path_img = os.path.join(best_match_dir, f'ID{pid:04d}_best.jpg')
            cv2.imwrite(save_path_img, data['img'])

        if vid_writer:
            vid_writer.release()
            self.loginfo(f"Results saved to {save_path}")
        self.loginfo(f'Done. ({time.time() - t0:.3f}s)')
        cv2.destroyAllWindows()


