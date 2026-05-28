"""
CV 球员跟踪模块
YOLOv8 检测 + ByteTrack 跟踪 + 球衣颜色分类 + 投篮者识别
"""
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv


class PlayerTracker:
    """篮球球员检测与跟踪"""

    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = YOLO(model_name)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.4,
            lost_track_buffer=30,
            minimum_matching_threshold=0.7,
            frame_rate=30,
        )
        # 每个 track_id 的颜色历史 (用于球衣颜色判定)
        self.color_history: dict[int, list[str]] = defaultdict(list)
        # 每个 track_id 的位置历史 [(frame_idx, center_x, center_y, bbox)]
        self.position_history: dict[int, list[tuple]] = defaultdict(list)

    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        """YOLOv8 检测所有人 (class=0)"""
        results = self.model(frame, verbose=False, classes=[0], conf=0.3)
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return sv.Detections.empty()

        # 转换为 supervision Detections 格式
        detections = sv.Detections.from_ultralytics(results[0])
        # 只保留高置信度的人体检测
        detections = detections[detections.confidence > 0.35]
        return detections

    def update_tracks(self, detections: sv.Detections) -> sv.Detections:
        """ByteTrack 更新跟踪"""
        return self.tracker.update_with_detections(detections)

    # HSV 预定义颜色范围 (排除肤色、阴影)
    SKIN_LOWER = np.array([0, 10, 40])
    SKIN_UPPER = np.array([25, 150, 255])
    DARK_UPPER = np.array([180, 50, 80])

    COLOR_RANGES = {
        "红色": ([(0, 50, 40), (10, 255, 255)],
                  [(160, 50, 40), (180, 255, 255)]),
        "橙色": ([(8, 50, 40), (25, 255, 255)],),
        "黄色": ([(22, 50, 40), (38, 255, 255)],),
        "绿色": ([(35, 50, 40), (85, 255, 255)],),
        "蓝色": ([(90, 50, 40), (130, 255, 255)],),
        "紫色": ([(130, 50, 40), (160, 255, 255)],),
        "白色": ([(0, 0, 180), (180, 40, 255)],),
    }

    def extract_jersey_color(self, frame: np.ndarray, bbox: np.ndarray) -> str:
        """HSV 颜色范围匹配球衣颜色，排除肤色和阴影"""
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return "unknown"

        person = frame[y1:y2, x1:x2]
        th, tw = person.shape[:2]
        # 取上半身靠中心的 60% 区域（排除边缘背景、短裤、手臂皮肤）
        top, bot = int(th * 0.1), int(th * 0.55)
        left, right = int(tw * 0.2), int(tw * 0.8)
        roi = person[top:bot, left:right]
        if roi.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        skin = cv2.inRange(hsv, self.SKIN_LOWER, self.SKIN_UPPER)
        dark = cv2.inRange(hsv, np.array([0, 0, 0]), self.DARK_UPPER)

        scores = {}
        total = roi.shape[0] * roi.shape[1]
        for name, ranges in self.COLOR_RANGES.items():
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask[skin > 0] = 0
            mask[dark > 0] = 0
            ratio = np.sum(mask > 0) / total
            if ratio > 0.06:
                scores[name] = ratio

        if not scores:
            return "unknown"
        return max(scores, key=scores.get)

    def process_goal_clip(self, video_path: Path, goal_time_sec: float,
                          window: float = 3.0) -> dict:
        """处理单个进球片段：跟踪球员 → 识别投篮者

        Args:
            video_path: 原始视频路径
            goal_time_sec: 进球时间戳 (秒)
            window: 分析窗口 ±N 秒

        Returns: {
            "scorer_jersey_color": str,
            "scorer_jersey_number_visible": bool,
            "all_players": [{track_id, jersey_color, positions}],
            "frames_processed": int,
        }
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        goal_frame = int(goal_time_sec * fps)
        start_frame = max(0, goal_frame - int(window * fps))
        end_frame = min(total_frames, goal_frame + int(window * fps))

        # 跳到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        self.color_history.clear()
        self.position_history.clear()

        frames_processed = 0
        # 跟踪每个球员在进球时刻前后的活跃度
        activity_at_goal: dict[int, float] = defaultdict(float)

        for frame_idx in range(start_frame, end_frame, 2):  # 每2帧处理一次
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect_players(frame)
            if len(detections) == 0:
                frames_processed += 1
                continue

            tracked = self.update_tracks(detections)

            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                if tid < 0:
                    continue

                bbox = tracked.xyxy[i]
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                color = self.extract_jersey_color(frame, bbox)

                self.color_history[tid].append(color)
                self.position_history[tid].append(
                    (frame_idx, center_x, center_y, bbox)
                )

                # 靠近进球帧的球员获得更高活动权重
                dist_to_goal = abs(frame_idx - goal_frame)
                if dist_to_goal < fps:  # 1 秒内
                    activity_at_goal[tid] += 3
                elif dist_to_goal < fps * 2:  # 2 秒内
                    activity_at_goal[tid] += 1

            frames_processed += 1

        cap.release()

        # ── 确定投篮者 ──
        # 策略：进球时刻最活跃 + 位置最接近篮筐的球员
        scorer_id = self._identify_scorer(activity_at_goal, goal_frame)
        scorer_color = self._dominant_color(scorer_id) if scorer_id else "unknown"

        # ── 汇总所有球员 ──
        all_players = []
        for tid in self.color_history:
            all_players.append({
                "track_id": tid,
                "jersey_color": self._dominant_color(tid),
                "color_confidence": self._color_confidence(tid),
                "is_scorer": tid == scorer_id,
                "frames_tracked": len(self.position_history[tid]),
            })

        return {
            "scorer_jersey_color": scorer_color,
            "scorer_track_id": scorer_id,
            "all_players": all_players,
            "frames_processed": frames_processed,
            "fps": fps,
            "goal_frame": goal_frame,
        }

    def _identify_scorer(self, activity: dict[int, float],
                         goal_frame: int) -> int | None:
        """确定投篮者：进球帧附近位置最高的球员（跳投/上篮时身体最高）"""
        if not self.position_history:
            return None

        # 在进球帧 ±0.5s 内，找出平均 y 坐标最小（位置最高）的球员
        scorer_scores = {}
        for tid, positions in self.position_history.items():
            near_goal = [(fi, cy) for fi, cx, cy, _ in positions
                         if abs(fi - goal_frame) < 15]
            if not near_goal:
                continue
            # 得分 = 活跃度权重 + 位置高度奖励（y 越小分越高）
            avg_y = sum(cy for _, cy in near_goal) / len(near_goal)
            scorer_scores[tid] = activity.get(tid, 1) + (200 - avg_y) * 0.5

        if not scorer_scores:
            # fallback: 任何帧中位置最高的球员
            for tid, positions in self.position_history.items():
                avg_y = sum(cy for _, cx, cy, _ in positions) / len(positions)
                scorer_scores[tid] = 200 - avg_y

        return max(scorer_scores, key=scorer_scores.get) if scorer_scores else None

    def _dominant_color(self, track_id: int) -> str:
        """球员的主导球衣颜色"""
        colors = self.color_history.get(track_id, [])
        if not colors:
            return "unknown"
        # 排除 unknown 后取众数
        known = [c for c in colors if c != "unknown"]
        if not known:
            return "unknown"
        return max(set(known), key=known.count)

    def _color_confidence(self, track_id: int) -> float:
        """颜色判定置信度"""
        colors = self.color_history.get(track_id, [])
        if not colors:
            return 0.0
        known = [c for c in colors if c != "unknown"]
        if not known:
            return 0.0
        dominant = self._dominant_color(track_id)
        return known.count(dominant) / len(known)
