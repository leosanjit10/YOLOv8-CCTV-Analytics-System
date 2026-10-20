"""YOLOv8 inference and model-device selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None = None

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))


class Detector:
    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.35,
        iou: float = 0.5,
        image_size: int = 640,
        device: str = "auto",
        classes: tuple[int, ...] | None = None,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.device = self._select_device(device)
        self.classes = list(classes) if classes else None

    @staticmethod
    def _select_device(requested: str) -> str:
        if requested.lower() == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        if requested != "cpu" and requested.isdigit() and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {requested} requested, but CUDA is unavailable")
        return requested

    def track_frame(self, frame: Any, tracker_config: str = "bytetrack.yaml") -> list[Detection]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=tracker_config,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        coordinates = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        track_ids = (
            boxes.id.int().cpu().tolist()
            if boxes.id is not None
            else [None] * len(coordinates)
        )
        names = result.names
        return [
            Detection(
                box=tuple(int(value) for value in coordinates[index]),
                class_id=class_id,
                class_name=str(names[class_id]),
                confidence=float(confidences[index]),
                track_id=track_ids[index],
            )
            for index, class_id in enumerate(class_ids)
        ]
