# -------------------- 导入库 --------------------
import time  # 时间控制，例如延时、获取时间戳
import os  # 文件和目录操作
import cv2  # OpenCV，用于视频读取、图像处理和绘制
import torch  # PyTorch，用于模型推理
import numpy as np  # 数值计算
from PIL import Image  # 图像处理
from ultralytics import YOLO  # YOLOv8 检测模型
from utils.extracter import FeatureExtractor  # 自定义 ReID 特征提取器
from utils.plots import plot_one_box  # 绘制检测框
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度计算
from numpy import random  # 随机数生成
import threading  # 多线程
import queue  # 队列，用于线程间传递数据
from scipy.optimize import linear_sum_assignment  # 匈牙利算法，用于最优匹配

# -------------------- 主类定义 --------------------
class MultiPersonSearcher:
    """
    多摄像头多人 ReID 搜索器
    支持：
        - YOLOv8 检测行人
        - 提取 ReID 特征
        - 跨摄像头匹配统一 Global ID
        - overlap 区域过滤（可选）
        - 匹配结果可视化与保存
    """
    def __init__(self,
                 weights='yolo11m.pt',           # YOLOv8 权重路径
                 reid_config='models/opts.yaml', # ReID 特征提取器配置
                 device='0' if torch.cuda.is_available() else 'cpu',  # 设备
                 img_size=640,                   # YOLO 输入尺寸
                 conf_thres=0.25,                # YOLO 检测置信度阈值
                 match_threshold=0.7,            # ReID 匹配阈值
                 save_dir='matched_frames',      # 匹配结果保存目录
                 overlap_area=None,              # 重叠区域定义
                 draw_overlap=True):             # 是否绘制 overlap 区域
        # -------------------- 初始化参数 --------------------
        self.device = device
        self.conf_thres = conf_thres
        self.match_threshold = match_threshold
        self.img_size = img_size
        self.draw_overlap = draw_overlap

        # 初始化 YOLO 模型
        self.model = YOLO(weights)
        self.names = self.model.names  # 类别名
        # 随机颜色列表，用于标注不同 ID
        self.colors_list = [[random.randint(0, 255) for _ in range(3)] for _ in range(1000)]
        if 'person' not in self.names.values():
            raise ValueError("YOLO 模型不支持 'person' 类别，请使用 COCO 预训练模型")

        # 初始化 ReID 特征提取器
        self.extractor = FeatureExtractor(reid_config)

        # 全局 ID 计数
        self.next_global_id = 1
        # 存储全局 ID 对应的特征向量
        self.target_features = {}  # {global_id: feature_vector}
        self.saved_global_ids = set()  # 避免重复保存匹配帧

        # 保存匹配结果的目录
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 缓存最新帧的队列，每个摄像头一个队列
        self.latest_frames = [queue.Queue(maxsize=1) for _ in range(2)]
        self.latest_processed_frame = None

        # overlap 区域定义
        self.overlap_area = overlap_area

    # -------------------- 辅助函数 --------------------
    def normalize_features(self, features):
        """L2 归一化特征向量"""
        norm = np.linalg.norm(features)
        return features / norm if norm > 0 else features

    def in_overlap_area_center(self, x1, x2, frame_width, cam_idx):
        """判断框中心是否在 overlap 区域"""
        if self.overlap_area is None:
            return True
        x_min = self.overlap_area[cam_idx]['x_min'] * frame_width
        x_max = self.overlap_area[cam_idx]['x_max'] * frame_width
        x_center = (x1 + x2) / 2
        return x_min <= x_center <= x_max
    
    def in_overlap_area(self, x1, x2, y1, y2, frame_width, frame_height, cam_idx):
        """判断人的框是否与 overlap 区域有交集"""
        if self.overlap_area is None:
            return True
        x_min = self.overlap_area[cam_idx]['x_min'] * frame_width
        x_max = self.overlap_area[cam_idx]['x_max'] * frame_width
        inter_x1 = max(x1, x_min)
        inter_x2 = min(x2, x_max)
        return inter_x2 > inter_x1  # 只要有交集就返回 True

    def draw_dashed_rect(self,img, pt1, pt2, color=(0,255,0), thickness=2, dash_length=10):
        """绘制虚线矩形"""
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

    # -------------------- 视频读取线程 --------------------
    def rtsp_reader(self, src, cam_idx, frame_size):
        """
        RTSP 视频流读取线程，将最新帧放入队列
        src: 视频源
        cam_idx: 摄像头索引
        frame_size: 输出帧尺寸
        """
        cap = cv2.VideoCapture(src)
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # 读取失败，返回黑帧
                frame = np.zeros((frame_size[1], frame_size[0], 3), np.uint8)
            else:
                frame = cv2.resize(frame, frame_size)
            if not self.latest_frames[cam_idx].full():
                self.latest_frames[cam_idx].put(frame)
            time.sleep(0.01)

    def video_reader(self, src, cam_idx, frame_size):
        """
        本地视频读取线程，循环读取
        """
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"无法打开视频源: {src}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 25
        frame_interval = 1.0 / fps

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if isinstance(src, str) and src.endswith((".mp4", ".avi")):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    frame = np.zeros((frame_size[1], frame_size[0], 3), np.uint8)
            else:
                frame = cv2.resize(frame, frame_size)

            if not self.latest_frames[cam_idx].full():
                self.latest_frames[cam_idx].put(frame)

            time.sleep(frame_interval)

    # -------------------- 异步处理 --------------------
    def processing_worker(self):
        """
        异步处理线程：
        - 获取最新帧
        - 调用核心处理函数
        - 更新 latest_processed_frame
        """
        while True:
            frames = [q.get() for q in self.latest_frames]
            if all(frame is not None for frame in frames):
                concat_frame = self.process_frames_and_draw(frames)
                self.latest_processed_frame = concat_frame
            time.sleep(0.01)

    # -------------------- 核心处理 --------------------
    def process_frames_and_draw(self, frames):
        """
        核心处理逻辑：
        1. YOLO 检测
        2. overlap 区域过滤
        3. ReID 特征提取
        4. 同帧去重
        5. 分配/更新 Global ID
        6. 跨摄像头匹配
        7. 绘制框 & 保存匹配帧
        """
        all_box_coords, all_global_ids, all_features, processed_frames = [], [], [], []

        for cam_idx, im0 in enumerate(frames):
            crops, box_coords, features = [], [], []

            # 绘制 overlap 区域
            if self.overlap_area and self.draw_overlap:
                x_min = int(self.overlap_area[cam_idx]['x_min'] * im0.shape[1])
                x_max = int(self.overlap_area[cam_idx]['x_max'] * im0.shape[1])
                self.draw_dashed_rect(im0, (x_min, 0), (x_max, im0.shape[0]), color=(0,255,0), thickness=2, dash_length=10)

            # YOLO 检测
            results = self.model(im0, imgsz=self.img_size, verbose=False)[0]
            if results.boxes is not None:
                boxes_xyxy, confs = [], []
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    # 只保留 person 且置信度高于阈值
                    if self.names[cls_id] != 'person' or conf < self.conf_thres:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h, w = im0.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    # overlap 区域过滤
                    if not self.in_overlap_area(x1, x2, y1, y2, w, h, cam_idx):
                        continue
                    boxes_xyxy.append([x1, y1, x2 - x1, y2 - y1])
                    confs.append(conf)

                # NMS 去重
                indices = cv2.dnn.NMSBoxes(boxes_xyxy, confs, self.conf_thres, 0.5)
                kept_boxes = [boxes_xyxy[i] for i in indices.flatten()] if len(indices) > 0 else []

                # 提取 ReID 特征
                for (x, y, w, h) in kept_boxes:
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    person_img = im0[y1:y2, x1:x2]
                    if person_img.size == 0:
                        continue
                    person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)).resize((128, 256))
                    crops.append(person_pil)
                    box_coords.append((x1, y1, x2, y2))

            if crops:
                features = [self.normalize_features(self.extractor.extract_feature(img)) for img in crops]

            # 同帧去重 (IoU > 0.7)
            def iou(box1, box2):
                x1, y1, x2, y2 = box1
                x1b, y1b, x2b, y2b = box2
                inter_x1, inter_y1 = max(x1, x1b), max(y1, y1b)
                inter_x2, inter_y2 = min(x2, x2b), min(y2, y2b)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (x2b - x1b) * (y2b - y1b)
                return inter_area / float(area1 + area2 - inter_area + 1e-6)

            keep_idx = list(range(len(box_coords)))
            removed = set()
            for i in range(len(box_coords)):
                if i in removed:
                    continue
                for j in range(i + 1, len(box_coords)):
                    if j in removed:
                        continue
                    if iou(box_coords[i], box_coords[j]) > 0.7:
                        removed.add(j)
            keep_idx = [i for i in keep_idx if i not in removed]

            crops = [crops[i] for i in keep_idx]
            box_coords = [box_coords[i] for i in keep_idx]
            features = [features[i] for i in keep_idx]

            # 分配 Global ID
            assigned_global_ids = []
            for feat in features:
                if self.target_features:
                    sims = cosine_similarity([feat], list(self.target_features.values()))[0]
                    max_idx = sims.argmax()
                    if sims[max_idx] >= self.match_threshold:
                        global_id = list(self.target_features.keys())[max_idx]
                    else:
                        global_id = self.next_global_id
                        self.next_global_id += 1
                else:
                    global_id = self.next_global_id
                    self.next_global_id += 1
                self.target_features[global_id] = feat
                assigned_global_ids.append(global_id)

            processed_frames.append(im0)
            all_box_coords.append(box_coords)
            all_global_ids.append(assigned_global_ids)
            all_features.append([self.target_features[gid] for gid in assigned_global_ids])

        # 跨摄像头匹配
        cross_sim_dict = {}
        if len(all_features) >= 2 and all_features[0] and all_features[1]:
            feats_left, ids_left = all_features[0], all_global_ids[0]
            feats_right, ids_right = all_features[1], all_global_ids[1]
            sim_matrix = cosine_similarity(feats_left, feats_right)
            row_ind, col_ind = linear_sum_assignment(-sim_matrix)  # 最大匹配
            for i, j in zip(row_ind, col_ind):
                sim = sim_matrix[i, j]
                if sim >= self.match_threshold:
                    gid_left, gid_right = ids_left[i], ids_right[j]
                    unified_id = min(gid_left, gid_right)  # 统一 ID
                    ids_left[i] = unified_id
                    ids_right[j] = unified_id
                    self.target_features[unified_id] = feats_left[i]
                    cross_sim_dict[unified_id] = sim

            all_global_ids[0] = list(ids_left)
            all_global_ids[1] = list(ids_right)

        # 绘制检测框
        concat_frame = np.hstack(processed_frames)
        for cam_idx, (box_coords, global_ids) in enumerate(zip(all_box_coords, all_global_ids)):
            x_offset = cam_idx * processed_frames[0].shape[1]  # 横向拼接偏移
            for idx, (x1, y1, x2, y2) in enumerate(box_coords):
                if idx >= len(global_ids):
                    continue
                gid = global_ids[idx]
                color = self.colors_list[gid % len(self.colors_list)]
                sim_label = f' sim:{cross_sim_dict[gid]:.2f}' if gid in cross_sim_dict else ''
                label = f'ID:{gid:04d}{sim_label}'
                plot_one_box([x1 + x_offset, y1, x2 + x_offset, y2],
                             concat_frame, label=label, color=color, line_thickness=3)

        # 保存匹配结果
        frame_time = int(time.time() * 1000)
        for gid, sim in cross_sim_dict.items():
            if gid in self.saved_global_ids:
                continue
            save_name = f"Cross_ID{gid:04d}_sim{sim:.2f}_{frame_time}.jpg"
            save_path = os.path.join(self.save_dir, save_name)
            cv2.imwrite(save_path, concat_frame)
            self.saved_global_ids.add(gid)

        return concat_frame

    # -------------------- 异步搜索 --------------------
    def search_videos_async(self, sources, view_img=True, window_size=(1280, 720)):
        """
        多摄像头异步搜索入口
        """
        num_cams = len(sources)
        frame_size = (window_size[0] // num_cams, window_size[1])

        # 启动每个摄像头视频读取线程
        for idx, src in enumerate(sources):
            threading.Thread(target=self.video_reader, args=(src, idx, frame_size), daemon=True).start()
        # 启动异步处理线程
        threading.Thread(target=self.processing_worker, daemon=True).start()

        # 创建显示窗口
        cv2.namedWindow('Multi-Camera ReID', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Multi-Camera ReID', *window_size)

        while True:
            if self.latest_processed_frame is not None:
                cv2.imshow('Multi-Camera ReID', self.latest_processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


# -------------------- 主函数 --------------------
def main():
    sources = [
        'files/video/camera_a.mp4',
        'files/video/camera_b.mp4'
    ]

    # 定义摄像头重叠区域
    overlap_area = [
        {'x_min': 0.2, 'x_max': 0.4},  # 左摄像头右侧 20%-40%
        {'x_min': 0.3, 'x_max': 0.7}   # 右摄像头左侧 30%-70%
    ]

    searcher = MultiPersonSearcher(
        weights='models/yolo11m.pt',
        reid_config='models/opts.yaml',
        device='0' if torch.cuda.is_available() else 'cpu',
        img_size=640,
        conf_thres=0.25,
        match_threshold=0.75,
        save_dir='matched_frames_mp4',
        overlap_area=overlap_area,
        draw_overlap=True
    )
    # 启动异步搜索
    searcher.search_videos_async(sources, view_img=True, window_size=(1920, 540))


if __name__ == '__main__':
    main()
