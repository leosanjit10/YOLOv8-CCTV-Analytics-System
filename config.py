"""Application settings loaded from JSON and environment variables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    model_path: Path = ROOT_DIR / "yolov8n.pt"
    source: str = "marathon.mp4"
    output_dir: Path = ROOT_DIR / "output"
    database_path: Path = ROOT_DIR / "data" / "analytics.db"
    confidence: float = 0.35
    iou: float = 0.5
    image_size: int = 640
    device: str = "auto"
    classes: tuple[int, ...] | None = None
    line_start: tuple[float, float] = (0.0, 0.5)
    line_end: tuple[float, float] = (1.0, 0.5)
    zones: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    reconnect_attempts: int = 5
    reconnect_delay_seconds: float = 2.0
    log_level: str = "INFO"
    log_file: Path = ROOT_DIR / "output" / "analytics.log"

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "AppConfig":
        """Load optional JSON settings, then let environment variables override them."""
        values: dict[str, Any] = {}
        path = Path(config_path) if config_path else ROOT_DIR / "config.json"
        if path.is_file():
            with path.open(encoding="utf-8") as config_file:
                values = json.load(config_file)
            if not isinstance(values, dict):
                raise ValueError(f"Configuration must be a JSON object: {path}")

        env_names = {
            "model_path": "ANALYTICS_MODEL",
            "source": "ANALYTICS_SOURCE",
            "output_dir": "ANALYTICS_OUTPUT_DIR",
            "database_path": "ANALYTICS_DATABASE",
            "confidence": "ANALYTICS_CONFIDENCE",
            "device": "ANALYTICS_DEVICE",
            "log_level": "ANALYTICS_LOG_LEVEL",
            "log_file": "ANALYTICS_LOG_FILE",
        }
        for key, env_name in env_names.items():
            if env_name in os.environ:
                values[key] = os.environ[env_name]

        path_fields = {"model_path", "output_dir", "database_path", "log_file"}
        for key in path_fields & values.keys():
            candidate = Path(values[key]).expanduser()
            values[key] = candidate if candidate.is_absolute() else ROOT_DIR / candidate

        if "classes" in values and values["classes"] is not None:
            values["classes"] = tuple(int(class_id) for class_id in values["classes"])
        for key in ("line_start", "line_end"):
            if key in values:
                values[key] = tuple(float(value) for value in values[key])
        if "zones" in values:
            values["zones"] = {
                name: [tuple(float(value) for value in point) for point in polygon]
                for name, polygon in values["zones"].items()
            }

        config = cls(**values)
        if not 0.0 <= config.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= config.iou <= 1.0:
            raise ValueError("iou must be between 0 and 1")
        if len(config.line_start) != 2 or len(config.line_end) != 2:
            raise ValueError("line_start and line_end must each contain two coordinates")
        if config.line_start == config.line_end:
            raise ValueError("line_start and line_end must be different points")
        for point in (config.line_start, config.line_end):
            if any(not 0.0 <= coordinate <= 1.0 for coordinate in point):
                raise ValueError("Line coordinates must be normalized between 0 and 1")
        for name, polygon in config.zones.items():
            if len(polygon) < 3:
                raise ValueError(f"Zone {name!r} must contain at least three points")
            if any(
                not 0.0 <= coordinate <= 1.0
                for point in polygon
                for coordinate in point
            ):
                raise ValueError(f"Zone {name!r} coordinates must be between 0 and 1")
        if config.image_size <= 0:
            raise ValueError("image_size must be greater than zero")
        if config.reconnect_attempts < 0 or config.reconnect_delay_seconds < 0:
            raise ValueError("Reconnect attempts and delay cannot be negative")
        return config
