"""Managed MP4 output for annotated and heatmap-overlay video."""

from __future__ import annotations

from pathlib import Path

import cv2


class VideoWriter:
    def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_fps = fps if fps > 0 else 25.0
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), safe_fps, frame_size
        )
        if not self.writer.isOpened():
            raise OSError(f"Could not create video output: {path}")

    def write(self, frame) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
