"""Detection-density accumulation and heatmap artifact generation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from detector import Detection


class HeatmapAnalytics:
    def __init__(self) -> None:
        self._density: np.ndarray | None = None
        self._movement: np.ndarray | None = None
        self._last_centers: dict[int, tuple[int, int]] = {}

    def update(self, frame_shape: tuple[int, ...], detections: list[Detection]) -> None:
        height, width = frame_shape[:2]
        if self._density is None or self._density.shape != (height, width):
            self._density = np.zeros((height, width), dtype=np.float32)
            self._movement = np.zeros((height, width), dtype=np.float32)
            self._last_centers.clear()
        for detection in detections:
            cv2.circle(self._density, detection.center, 12, 1.0, thickness=-1)
            if detection.track_id is not None:
                previous = self._last_centers.get(detection.track_id)
                if previous is not None:
                    cv2.line(
                        self._movement, previous, detection.center, 1.0, thickness=3
                    )
                self._last_centers[detection.track_id] = detection.center

    def overlay(self, frame: np.ndarray, alpha: float = 0.38) -> np.ndarray:
        if self._density is None or not np.any(self._density):
            return frame
        combined = self._density
        if self._movement is not None:
            combined = combined + self._movement
        normalized = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized.astype(np.uint8), (0, 0), 15)
        color = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1.0 - alpha, color, alpha, 0)

    def save(self, image_path: Path) -> None:
        self._save_layer(self._density, image_path)

    def save_movement(self, image_path: Path) -> None:
        self._save_layer(self._movement, image_path)

    @staticmethod
    def _save_layer(layer: np.ndarray | None, image_path: Path) -> None:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if layer is None or not np.any(layer):
            raise ValueError("No detections were available to generate a heatmap")
        normalized = cv2.normalize(layer, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        if not cv2.imwrite(str(image_path), heatmap):
            raise OSError(f"Could not write heatmap image: {image_path}")
