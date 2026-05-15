from __future__ import annotations

import base64
from collections import Counter, deque
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass(slots=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str
    area_ratio: float


class TrafficDetector:
    """YOLO12-based spatial analyzer for traffic frames."""

    def __init__(
        self,
        model_path: str | Path = "models/yolo12n.pt",
        road_capacity: float = 100.0,
        aggregation_window_seconds: int = 300,
        confidence_threshold: float = 0.25,
    ) -> None:
        self.model_path = Path(model_path)
        self.road_capacity = road_capacity
        self.aggregation_window_seconds = aggregation_window_seconds
        self.confidence_threshold = confidence_threshold
        self.model: object | None = None
        self.buffer: deque[dict[str, object]] = deque()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is None:
            from ultralytics import YOLO

            load_target = str(self.model_path if self.model_path.exists() else "yolo12n.pt")
            self.model = YOLO(load_target)

    def analyze_base64(self, frame_base64: str, road_capacity: float | None = None) -> dict[str, object]:
        frame_bytes = base64.b64decode(frame_base64)
        return self.analyze_bytes(frame_bytes, road_capacity=road_capacity)

    def analyze_bytes(self, frame_bytes: bytes, road_capacity: float | None = None) -> dict[str, object]:
        image = self._decode_image(frame_bytes)
        return self.analyze_image(image, road_capacity=road_capacity)

    def analyze_video(
        self,
        video_path: str | Path,
        sample_every_seconds: float = 2.0,
        road_capacity: float | None = None,
        max_frames: int = 60,
    ) -> dict[str, object]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("Unable to open video file.")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        sample_every_frames = max(int(fps * sample_every_seconds), 1)
        frame_index = 0
        sampled = []

        try:
            while len(sampled) < max_frames:
                success, frame = capture.read()
                if not success:
                    break

                if frame_index % sample_every_frames == 0:
                    analysis = self.analyze_image(frame, road_capacity=road_capacity)
                    sampled.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_seconds": float(frame_index / fps),
                            "analysis": analysis,
                        }
                    )
                frame_index += 1
        finally:
            capture.release()

        vehicle_counts = [int(item["analysis"]["total_vehicles"]) for item in sampled]
        chaos_values = [float(item["analysis"]["chaos_index"]) for item in sampled]
        return {
            "sampled_frames": len(sampled),
            "average_vehicle_count": float(np.mean(vehicle_counts)) if vehicle_counts else 0.0,
            "max_vehicle_count": max(vehicle_counts) if vehicle_counts else 0,
            "average_chaos_index": float(np.mean(chaos_values)) if chaos_values else 0.0,
            "frames": sampled,
        }

    def analyze_image(self, image: np.ndarray, road_capacity: float | None = None) -> dict[str, object]:
        self.load()
        assert self.model is not None

        height, width = image.shape[:2]
        frame_area = max(float(height * width), 1.0)
        result = self.model.predict(image, verbose=False, conf=self.confidence_threshold)[0]
        detections = self._extract_vehicle_detections(result, frame_area)
        counts = Counter(detection.label for detection in detections)
        total_vehicles = len(detections)
        average_area = float(np.mean([item.area_ratio for item in detections])) if detections else 0.0
        capacity = float(road_capacity or self.road_capacity)
        chaos_index = (total_vehicles * average_area) / capacity
        timestamp = datetime.now(timezone.utc)

        analysis = {
            "total_vehicles": total_vehicles,
            "counts": {label: counts.get(label, 0) for label in VEHICLE_CLASSES.values()},
            "chaos_index": chaos_index,
            "average_area_occupied": average_area,
            "boxes": [asdict(detection) for detection in detections],
            "timestamp": timestamp,
        }
        self._append_aggregation(analysis)
        return analysis

    def get_recent_aggregates(self) -> list[dict[str, object]]:
        self._evict_old_entries()
        return list(self.buffer)

    def five_minute_summary(self) -> dict[str, float]:
        self._evict_old_entries()
        if not self.buffer:
            return {
                "vehicle_count": 0.0,
                "chaos_index": 0.0,
                "average_area_occupied": 0.0,
            }

        return {
            "vehicle_count": float(sum(int(item["total_vehicles"]) for item in self.buffer)),
            "chaos_index": float(np.mean([float(item["chaos_index"]) for item in self.buffer])),
            "average_area_occupied": float(np.mean([float(item["average_area_occupied"]) for item in self.buffer])),
        }

    def _append_aggregation(self, analysis: dict[str, object]) -> None:
        self.buffer.append(analysis)
        self._evict_old_entries()

    def _evict_old_entries(self) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - self.aggregation_window_seconds
        while self.buffer and self._timestamp(self.buffer[0]) < cutoff:
            self.buffer.popleft()

    @staticmethod
    def _timestamp(item: dict[str, object]) -> float:
        timestamp = item.get("timestamp")
        if isinstance(timestamp, datetime):
            return timestamp.timestamp()
        return 0.0

    @staticmethod
    def _decode_image(frame_bytes: bytes) -> np.ndarray:
        array = np.frombuffer(frame_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to decode image frame.")
        return image

    @staticmethod
    def _extract_vehicle_detections(result: object, frame_area: float) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections: list[Detection] = []
        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            if class_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            area_ratio = max((x2 - x1) * (y2 - y1), 0.0) / frame_area
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    label=VEHICLE_CLASSES[class_id],
                    area_ratio=area_ratio,
                )
            )
        return detections
