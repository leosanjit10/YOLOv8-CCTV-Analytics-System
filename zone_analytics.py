"""Polygon-zone occupancy, dwell-time, and traffic measurements."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

import cv2
import numpy as np

from detector import Detection


class ZoneAnalytics:
    def __init__(
        self,
        zones: dict[str, list[tuple[float, float]]],
        missing_track_grace_seconds: float = 2.0,
    ) -> None:
        self.zones = zones
        self.missing_track_grace_seconds = missing_track_grace_seconds
        self._inside: set[tuple[int, str]] = set()
        self._entered_at: dict[tuple[int, str], datetime] = {}
        self._last_seen: dict[tuple[int, str], datetime] = {}
        self._last_detection: dict[tuple[int, str], Detection] = {}
        self.occupancy: Counter[str] = Counter()
        self.traffic: Counter[str] = Counter()
        self.dwell_seconds: defaultdict[str, float] = defaultdict(float)

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, ...],
        timestamp: datetime,
    ) -> list[tuple[Detection, str, str, float]]:
        height, width = frame_shape[:2]
        events: list[tuple[Detection, str, str, float]] = []
        active_track_ids = {
            detection.track_id
            for detection in detections
            if detection.track_id is not None
        }
        current = {
            (detection.track_id, name): detection
            for detection in detections
            if detection.track_id is not None
            for name, polygon in self.zones.items()
            if cv2.pointPolygonTest(
                np.asarray(
                    [(int(x * width), int(y * height)) for x, y in polygon],
                    dtype=np.int32,
                ),
                detection.center,
                False,
            )
            >= 0
        }

        for key, detection in current.items():
            object_id, zone_name = key
            if key not in self._inside:
                self._entered_at[key] = timestamp
                self.occupancy[zone_name] += 1
                self.traffic[zone_name] += 1
                events.append((detection, zone_name, "ZONE_ENTER", 0.0))
            self._last_seen[key] = timestamp
            self._last_detection[key] = detection

        retained = self._inside | current.keys()
        for key in self._inside - current.keys():
            object_id, zone_name = key
            last_seen = self._last_seen[key]
            if (
                object_id not in active_track_ids
                and (timestamp - last_seen).total_seconds()
                <= self.missing_track_grace_seconds
            ):
                continue
            entered_at = self._entered_at.pop(key)
            self._last_seen.pop(key)
            duration = max(0.0, (timestamp - entered_at).total_seconds())
            self.occupancy[zone_name] = max(0, self.occupancy[zone_name] - 1)
            self.dwell_seconds[zone_name] += duration
            events.append(
                (self._last_detection.pop(key), zone_name, "ZONE_EXIT", duration)
            )
            retained.discard(key)

        self._inside = set(retained)
        return events

    def most_visited(self) -> str | None:
        return self.traffic.most_common(1)[0][0] if self.traffic else None
