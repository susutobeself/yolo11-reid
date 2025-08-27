import time
import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from utils.extracter import FeatureExtractor
from utils.plots import plot_one_box
from sklearn.metrics.pairwise import cosine_similarity
from numpy import random
import threading
import queue
from scipy.optimize import linear_sum_assignment


class MultiPersonSearcher:
    """
    多摄像头多人体 ReID 系统，支持 YOLOv8-Pose 人体检测 + ReID 特征提取 + pose & color 特征融合。
    可跨摄像头统一 Global ID，并在 overlap 区域绘制虚线标识。
    """
    def __init__(self,
                 weights='models/yolov8n-pose.pt',    # YOLOv8-Pose 模型权重
                 reid_config='models/opts.yaml',  # ReID 模型配置文件
                 device='0' if torch.cuda.is_available() else 'cpu',  # 设备选择
                 img_size=640,                 # YOLO 输入图像大小
                 conf_thres=0.25,              # YOLO 检测置信度阈值
                 match_threshold=0.7,          # 融合特征匹配阈值
                 reid_weight=0.6,              # ReID 特征权重
                 pose_weight=0.3,              # Pose 特征权重
                 color_weight=0.1,             # 颜色直方图特征权重
                 reid_filter_thres=0.7,        # ReID 特征过滤阈值
                 save_dir='matched_frames',    # 保存匹配帧的文件夹
                 overlap_area=None,            # 重叠区域坐标列表 [{'x_min': , 'x_max': }, ...]
                 draw_overlap=False,            # 是否绘制 overlap 区域虚线
                 use_y_consistency=False):      # 是否使用 y 坐标中心一致性
                 
        # 初始化参数
        self.device = device
        self.conf_thres = conf_thres
        self.match_threshold = match_threshold
        self.reid_weight = reid_weight
        self.pose_weight = pose_weight
        self.color_weight = color_weight
        self.reid_filter_thres = reid_filter_thres
        self.img_size = img_size
        self.overlap_area = overlap_area
        self.draw_overlap = draw_overlap
        self.use_y_consistency = use_y_consistency

        # YOLOv8-Pose 模型
        self.model = YOLO(weights)
        self.names = self.model.names  # 类别名称字典
        # 随机生成颜色列表，用于绘制不同 Global ID 的框
        self.colors_list = [[random.randint(0, 255) for _ in range(3)] for _ in range(1000)]
        if 'person' not in self.names.values():
            raise ValueError("YOLO 模型不支持 'person' 类别，请使用 COCO 预训练模型")

        # ReID 特征提取器
        self.extractor = FeatureExtractor(reid_config)
        self.target_features = {}        # 保存每个 global_id 的特征字典
        self.next_global_id = 1          # 下一个分配的 Global ID
        self.saved_global_ids = set()    # 已保存的 Global ID，用于去重保存
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 视频帧队列
        # 每个摄像头最新一帧缓存队列（异步线程用）
        self.latest_frames = [queue.Queue(maxsize=1) for _ in range(2)]
        self.latest_processed_frame = None  # 最后处理后的多摄像头拼接帧


    @staticmethod
    def normalize_features(features, dim=None):
        """
        特征归一化函数
        """
        if features is None or len(features) == 0:
            return np.zeros(dim, dtype=np.float32) if dim else np.array([])
        norm = np.linalg.norm(features)
        return features / norm if norm > 0 else features

    def in_overlap_area_center(self, x1, x2, frame_width, cam_idx):
        """
        判断框中心是否在 overlap 区域
        """
        if self.overlap_area is None:
            return True
        x_min = self.overlap_area[cam_idx]['x_min'] * frame_width
        x_max = self.overlap_area[cam_idx]['x_max'] * frame_width
        x_center = (x1 + x2) / 2
        return x_min <= x_center <= x_max
    
    def in_overlap_area(self, x1, x2, y1, y2, frame_width, frame_height, cam_idx):
        """
        判断人的框是否与 overlap 区域有交集
        """
        if self.overlap_area is None:
            return True
        x_min = self.overlap_area[cam_idx]['x_min'] * frame_width
        x_max = self.overlap_area[cam_idx]['x_max'] * frame_width
        inter_x1 = max(x1, x_min)
        inter_x2 = min(x2, x_max)
        return inter_x2 > inter_x1  # 有交集即返回 True
    
    def draw_dashed_rect(self, img, pt1, pt2, color=(0,255,0), thickness=2, dash_length=10):
        """
        绘制虚线矩形
        """
        x1, y1 = pt1
        x2, y2 = pt2
        # 上下横线
        for x in range(x1, x2, 2*dash_length):
            x_end = min(x + dash_length, x2)
            cv2.line(img, (x, y1), (x_end, y1), color, thickness)
            cv2.line(img, (x, y2), (x_end, y2), color, thickness)
        # 左右竖线
        for y in range(y1, y2, 2*dash_length):
            y_end = min(y + dash_length, y2)
            cv2.line(img, (x1, y), (x1, y_end), color, thickness)
            cv2.line(img, (x2, y), (x2, y_end), color, thickness)

    @staticmethod
    def extract_pose_features(keypoints, box=None):
        """
        根据人体关键点提取 Pose 特征向量（14维）
        """
        if keypoints is None or len(keypoints) == 0:
            return np.zeros(14, dtype=np.float32)
        kp_coords = keypoints.xy[0].cpu().numpy()[:, :2]  # 提取坐标
        mean_xy = np.nanmean(kp_coords, axis=0)
        kp_coords = np.where(np.isnan(kp_coords), mean_xy, kp_coords)
        # 归一化到框内
        if box is not None:
            x1, y1, x2, y2 = box
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            kp_coords[:, 0] = (kp_coords[:, 0] - x1) / w
            kp_coords[:, 1] = (kp_coords[:, 1] - y1) / h
        # 选择稳定关键点
        stable_idx = [0, 1, 2, 3, 5, 6]
        kp_coords = kp_coords[stable_idx]
        neck = kp_coords[1]
        rel_coords = kp_coords - neck
        def angle(p1, p2):
            v = p2 - p1
            return np.arctan2(v[1], v[0])
        left_shoulder_angle = angle(kp_coords[2], kp_coords[5])
        left_hip_angle = angle(kp_coords[4], kp_coords[5])
        shoulder_width = np.linalg.norm(kp_coords[2] - kp_coords[3])
        hip_width = np.linalg.norm(kp_coords[4] - kp_coords[5])
        body_height = np.max(kp_coords[:, 1]) - np.min(kp_coords[:, 1])
        ratios = np.array([shoulder_width / body_height, hip_width / body_height], dtype=np.float32) if body_height > 1e-6 else np.array([0.0, 0.0], dtype=np.float32)
        vectors = np.concatenate([rel_coords.flatten(), [left_shoulder_angle, left_hip_angle], ratios])
        vectors = np.nan_to_num(vectors, nan=0.0)
        coord_norm = np.linalg.norm(vectors[:-4])
        if coord_norm > 0:
            vectors[:-4] /= coord_norm
        return vectors.astype(np.float32)

    @staticmethod
    def mirror_pose_feature(kp_coords):
        """
        镜像 Pose 特征（左右翻转）
        """
        mirror_idx = [0, 1, 3, 2, 5, 4]
        return kp_coords[mirror_idx]

    @staticmethod
    def extract_color_hist(img):
        """
        提取颜色直方图特征（HSV 三通道8x8x8）
        """
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hist = hist.flatten()
        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        return hist.astype(np.float32)

    def extract_part_based_feature(self, img: Image.Image):
        """
        分部位提取 ReID 特征（上/中/下三部分 + 全局特征融合）
        """
        img = img.resize((128, 256))
        w, h = img.size
        upper = img.crop((0, 0, w, h // 3))
        middle = img.crop((0, h // 3, w, 2 * h // 3))
        lower = img.crop((0, 2 * h // 3, w, h))
        # 提取特征并归一化
        feat_upper = self.normalize_features(self.extractor.extract_feature(upper))
        feat_middle = self.normalize_features(self.extractor.extract_feature(middle))
        feat_lower = self.normalize_features(self.extractor.extract_feature(lower))
        # 部位特征取最大值
        feat_part = np.maximum.reduce([feat_upper, feat_middle, feat_lower])
        # 全局特征
        feat_global = self.normalize_features(self.extractor.extract_feature(img))
        # 融合全局+部位
        feat_final = self.normalize_features(0.5 * feat_global + 0.5 * feat_part)
        return feat_final

    # 单帧处理
    def process_frame(self, im0, cam_idx=0):
        """
        处理单帧图像：
        - YOLO 人体检测
        - overlap 区域筛选
        - 提取 ReID / Pose / Color 特征
        """
        crops, box_coords, reid_feats, pose_feats, color_feats = [], [], [], [], []
        results = self.model(im0, imgsz=self.img_size, verbose=False)[0]

        # 绘制 overlap 区域虚线
        if self.overlap_area and self.draw_overlap:
            x_min = int(self.overlap_area[cam_idx]['x_min'] * im0.shape[1])
            x_max = int(self.overlap_area[cam_idx]['x_max'] * im0.shape[1])
            self.draw_dashed_rect(im0, (x_min, 0), (x_max, im0.shape[0]), (0, 255, 0), thickness=2, dash_length=15)

        # 遍历检测框和关键点
        for box, kps in zip(results.boxes or [], results.keypoints or [None]*len(results.boxes)):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if self.names[cls_id] != 'person' or conf < self.conf_thres:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = im0.shape[:2]
            # 筛选 overlap 区域
            if not self.in_overlap_area(x1, x2, y1, y2, w, h, cam_idx):
                continue
            person_img = im0[y1:y2, x1:x2]
            if person_img.size == 0:
                continue
            pil_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
            crops.append(pil_img)
            box_coords.append((x1, y1, x2, y2))
            reid_feats.append([self.extract_part_based_feature(pil_img),
                               self.extract_part_based_feature(pil_img.transpose(Image.FLIP_LEFT_RIGHT))])
            if kps is not None:
                pose_feats.append([self.extract_pose_features(kps, (x1, y1, x2, y2)),
                                   self.mirror_pose_feature(self.extract_pose_features(kps, (x1, y1, x2, y2)))])
            else:
                pose_feats.append([np.zeros(14), np.zeros(14)])
            color_feats.append(self.extract_color_hist(person_img))

        return crops, box_coords, reid_feats, pose_feats, color_feats

    # 多帧处理
    def process_frames(self, frames):
        """
        处理多摄像头帧：
        - 逐帧提取特征
        - 匹配 Global ID（跨摄像头）
        - 绘制 ID + 相似度
        - 保存匹配帧
        """
        all_box_coords, all_global_ids = [], []
        processed_frames = []
        features_list = []

        for cam_idx, im0 in enumerate(frames):
            crops, boxes, reid_feats, pose_feats, color_feats = self.process_frame(im0, cam_idx)
            assigned_ids = []

            # 逐目标匹配
            for r_feat, p_feat, c_feat in zip(reid_feats, pose_feats, color_feats):
                if self.target_features:
                    sims = []
                    for gid, feats in self.target_features.items():
                        reid_sim = max(cosine_similarity([r_feat[0]], [feats['reid'][0]])[0][0],
                                    cosine_similarity([r_feat[1]], [feats['reid'][1]])[0][0])
                        if reid_sim < self.reid_filter_thres:
                            sims.append(0.0)
                            continue
                        pose_sim = max(cosine_similarity([p_feat[0]], [feats['pose'][0]])[0][0],
                                    cosine_similarity([p_feat[1]], [feats['pose'][1]])[0][0])
                        color_sim = cosine_similarity([c_feat], [feats['color']])[0][0]
                        sims.append(self.reid_weight * reid_sim +
                                    self.pose_weight * pose_sim +
                                    self.color_weight * color_sim)
                    sims = np.array(sims)
                    max_idx = sims.argmax()
                    if sims[max_idx] >= self.match_threshold:
                        gid = list(self.target_features.keys())[max_idx]
                    else:
                        gid = self.next_global_id
                        self.next_global_id += 1
                else:
                    gid = self.next_global_id
                    self.next_global_id += 1
                # 更新特征字典
                self.target_features[gid] = {'reid': r_feat, 'pose': p_feat, 'color': c_feat}
                assigned_ids.append(gid)

            processed_frames.append(im0)
            all_box_coords.append(boxes)
            all_global_ids.append(assigned_ids)
            features_list.append(list(zip(reid_feats, pose_feats, color_feats)))

        # 跨摄像头匹配
        cross_sim_dict = {}
        if len(features_list) >= 2 and features_list[0] and features_list[1]:
            feats1, ids1, boxes1 = features_list[0], all_global_ids[0], all_box_coords[0]
            feats2, ids2, boxes2 = features_list[1], all_global_ids[1], all_box_coords[1]
            sim_matrix = np.zeros((len(feats1), len(feats2)), dtype=np.float32)
            h_frame = frames[0].shape[0]   ### 新增：取图像高度做归一化

            for i, (r1, p1, c1) in enumerate(feats1):
                for j, (r2, p2, c2) in enumerate(feats2):
                    reid_sim = max(cosine_similarity([r1[0]], [r2[0]])[0][0],
                                cosine_similarity([r1[1]], [r2[1]])[0][0])
                    if reid_sim < self.reid_filter_thres:
                        continue
                    pose_sim = max(cosine_similarity([p1[0]], [p2[0]])[0][0],
                                cosine_similarity([p1[1]], [p2[1]])[0][0])
                    color_sim = cosine_similarity([c1], [c2])[0][0]

                    # -------- y 坐标中心差值一致性 --------
                    y_center1 = (boxes1[i][1] + boxes1[i][3]) / 2
                    y_center2 = (boxes2[j][1] + boxes2[j][3]) / 2
                    y_diff = abs(y_center1 - y_center2)
                    y_overlap = 1 - (y_diff / h_frame)

                    # 最终分数 = 特征相似度 × y一致性
                    base_sim = self.reid_weight * reid_sim + self.pose_weight * pose_sim + self.color_weight * color_sim
                    if self.use_y_consistency:
                        final_sim = base_sim * y_overlap
                    else:
                        final_sim = base_sim

                    sim_matrix[i, j] = final_sim

                    ### 新增：打印调试信息
                    print(f"[跨摄像头候选] L{i}-R{j} | "
                        f"ReID={reid_sim:.3f}, Pose={pose_sim:.3f}, Color={color_sim:.3f}, "
                        f"y_diff={y_diff:.1f}, y_overlap={y_overlap:.3f}, Final={final_sim:.3f}")

            # 匈牙利算法分配 ID
            row_ind, col_ind = linear_sum_assignment(-sim_matrix)
            for i, j in zip(row_ind, col_ind):
                sim = sim_matrix[i, j]
                if sim >= self.match_threshold:
                    gid1, gid2 = ids1[i], ids2[j]
                    unified_id = min(gid1, gid2)
                    ids1[i] = unified_id
                    ids2[j] = unified_id
                    cross_sim_dict[unified_id] = sim
            all_global_ids[0], all_global_ids[1] = ids1, ids2

        # 绘制结果
        concat_frame = np.hstack(processed_frames)
        for cam_idx, (boxes, gids) in enumerate(zip(all_box_coords, all_global_ids)):
            x_offset = cam_idx * processed_frames[0].shape[1]
            for idx, (x1, y1, x2, y2) in enumerate(boxes):
                if idx >= len(gids):
                    continue
                gid = gids[idx]
                color = self.colors_list[gid % len(self.colors_list)]
                sim_label = f' sim:{cross_sim_dict[gid]:.2f}' if gid in cross_sim_dict else ''
                label = f'ID:{gid:04d}{sim_label}'
                plot_one_box([x1 + x_offset, y1, x2 + x_offset, y2], concat_frame, label=label,
                            color=color, line_thickness=3)

        # 保存匹配帧
        frame_time = int(time.time() * 1000)
        for gid, sim in cross_sim_dict.items():
            if gid in self.saved_global_ids:
                continue
            save_path = os.path.join(self.save_dir, f"Cross_ID{gid:04d}_sim{sim:.2f}_{frame_time}.jpg")
            cv2.imwrite(save_path, concat_frame)
            self.saved_global_ids.add(gid)

        return concat_frame


    # 视频读取线程
    def video_reader(self, src, cam_idx, frame_size):
        """
        异步读取视频帧并放入队列
        """
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"无法打开视频源: {src}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 and fps < 120 else 25
        frame_interval = 1.0 / fps
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if src.endswith(('.mp4', '.avi')):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    frame = np.zeros((frame_size[1], frame_size[0], 3), np.uint8)
            else:
                frame = cv2.resize(frame, frame_size)
            if not self.latest_frames[cam_idx].full():
                self.latest_frames[cam_idx].put(frame)
            time.sleep(frame_interval)

    # 异步处理线程
    def processing_worker(self):
        """
        从帧队列读取多摄像头帧并处理
        """
        while True:
            frames = [q.get() for q in self.latest_frames]
            if all(frame is not None for frame in frames):
                self.latest_processed_frame = self.process_frames(frames)
            time.sleep(0.01)

    # 异步视频搜索入口
    def search_videos_async(self, sources, view_img=True, window_size=(1280, 720)):
        """
        异步处理多个摄像头视频
        """
        num_cams = len(sources)
        frame_size = (window_size[0] // num_cams, window_size[1])
        # 启动视频读取线程
        for idx, src in enumerate(sources):
            threading.Thread(target=self.video_reader, args=(src, idx, frame_size), daemon=True).start()
        # 启动处理线程
        threading.Thread(target=self.processing_worker, daemon=True).start()

        cv2.namedWindow('Multi-Camera ReID', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Multi-Camera ReID', *window_size)

        while True:
            if self.latest_processed_frame is not None:
                cv2.imshow('Multi-Camera ReID', self.latest_processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def main():
    """
    多摄像头 ReID 测试入口
    """
    sources = ['files/video/camera_a.mp4', 'files/video/camera_b.mp4']
    overlap_area = [
        {'x_min': 0.2, 'x_max': 0.7},  # 左摄像头右侧 50% 区域
        {'x_min': 0.3, 'x_max': 0.4}   # 右摄像头左侧 10% 区域
    ]
    searcher = MultiPersonSearcher(
        weights='models/yolov8n-pose.pt',
        reid_config='models/opts.yaml',
        device='0' if torch.cuda.is_available() else 'cpu',
        img_size=640,
        conf_thres=0.25,
        match_threshold=0.75,
        reid_weight=0.6,
        pose_weight=0.3,
        color_weight=0.1,
        reid_filter_thres=0.7,
        save_dir='matched_frames_m5',
        # overlap_area=overlap_area,
        # draw_overlap=True,
        use_y_consistency=False
    )
    searcher.search_videos_async(sources, view_img=True, window_size=(1920, 540))


if __name__ == '__main__':
    main()
