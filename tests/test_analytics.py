from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import cv2

from app import VideoSource
from config import AppConfig
from counter import ObjectCounter
from database import AnalyticsDatabase
from detector import Detection
from heatmap import HeatmapAnalytics
from line_counter import LineCounter
from zone_analytics import ZoneAnalytics


def detection(x: int, y: int, object_id: int = 7) -> Detection:
    return Detection((x - 2, y - 2, x + 2, y + 2), 0, "person", 0.9, object_id)


class LineCounterTests(unittest.TestCase):
    def test_direction_and_duplicate_suppression(self) -> None:
        counter = LineCounter((0.0, 0.5), (1.0, 0.5))
        shape = (100, 200, 3)
        self.assertEqual(counter.update([detection(50, 75)], shape), [])
        entered = counter.update([detection(50, 25)], shape)
        self.assertEqual([direction for _, direction in entered], ["IN"])
        self.assertEqual(counter.update([detection(50, 20)], shape), [])
        exited = counter.update([detection(50, 75)], shape)
        self.assertEqual([direction for _, direction in exited], ["OUT"])
        self.assertEqual((counter.entries, counter.exits, counter.occupancy), (1, 1, 0))

    def test_deadband_does_not_reset_previous_side(self) -> None:
        counter = LineCounter((0.0, 0.5), (1.0, 0.5), deadband_pixels=4)
        shape = (100, 200, 3)
        counter.update([detection(50, 75)], shape)
        self.assertEqual(counter.update([detection(50, 50)], shape), [])
        self.assertEqual(
            [direction for _, direction in counter.update([detection(50, 25)], shape)],
            ["IN"],
        )


class ZoneAnalyticsTests(unittest.TestCase):
    def test_traffic_dwell_and_brief_missing_track(self) -> None:
        zone = ZoneAnalytics(
            {"entry": [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]},
            missing_track_grace_seconds=2,
        )
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        shape = (100, 100, 3)
        entered = zone.update([detection(20, 40)], shape, start)
        self.assertEqual(entered[0][2], "ZONE_ENTER")
        self.assertEqual(zone.occupancy["entry"], 1)
        self.assertEqual(zone.update([], shape, start + timedelta(seconds=1)), [])
        self.assertEqual(zone.occupancy["entry"], 1)
        exited = zone.update(
            [detection(80, 40)], shape, start + timedelta(seconds=3)
        )
        self.assertEqual(exited[0][0].class_name, "person")
        self.assertEqual(exited[0][2], "ZONE_EXIT")
        self.assertEqual(exited[0][3], 3.0)
        self.assertEqual(zone.dwell_seconds["entry"], 3.0)
        self.assertEqual(zone.traffic["entry"], 1)
        self.assertEqual(zone.occupancy["entry"], 0)


class CounterAndDatabaseTests(unittest.TestCase):
    def test_unique_and_per_class_counts(self) -> None:
        counter = ObjectCounter()
        counter.update([detection(10, 10), detection(20, 20), detection(30, 30, 8)])
        self.assertEqual(counter.total_detections, 3)
        self.assertEqual(counter.unique_count, 2)
        self.assertEqual(counter.unique_by_class["person"], 2)
        self.assertEqual(counter.detections_by_class["person"], 3)

    def test_session_summary_and_csv_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = AnalyticsDatabase(root / "events.db")
            session = database.start_session("clip.mp4")
            now = datetime.now(timezone.utc)
            database.log_event(detection(10, 10), "TRACK_STARTED", 1, timestamp=now, session_id=session)
            database.log_event(
                detection(10, 10), "LINE_CROSSING_IN", 5, "IN", now, session
            )
            database.log_event(
                detection(10, 10), "LINE_CROSSING_OUT", 8, "OUT", now, session
            )
            database.finish_session(session, 3, 1, 1, 1, 0)
            other_session = database.start_session("other.mp4")
            database.log_event(
                detection(10, 10), "TRACK_STARTED", 1, timestamp=now,
                session_id=other_session,
            )

            summary = database.summary(session)
            self.assertEqual(summary["entries"], 1)
            self.assertEqual(summary["exits"], 1)
            self.assertEqual(summary["occupancy"], 0)
            self.assertEqual(summary["class_counts"], {"person": 1})
            self.assertEqual(len(database.fetch_events(session_id=session)), 3)
            database.export_csv(root / "events.csv")
            with (root / "events.csv").open(encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                rows = list(reader)
                self.assertEqual(len(rows), 4)
                self.assertIn("session_id", reader.fieldnames)
                self.assertEqual(rows[0]["session_id"], session)

    def test_migrates_database_without_session_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy.db"
            import sqlite3

            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute(
                    """CREATE TABLE events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        object_id INTEGER,
                        object_class TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        direction TEXT,
                        frame_number INTEGER NOT NULL
                    )"""
                )
            database = AnalyticsDatabase(path)
            self.assertEqual(database.fetch_events(), [])


class HeatmapAndConfigurationTests(unittest.TestCase):
    def test_heatmap_layers_write_images(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary_directory:
            heatmap = HeatmapAnalytics()
            heatmap.update((100, 100, 3), [detection(20, 20)])
            heatmap.update((100, 100, 3), [detection(40, 40)])
            density_path = Path(temporary_directory) / "density.png"
            movement_path = Path(temporary_directory) / "movement.png"
            heatmap.save(density_path)
            heatmap.save_movement(movement_path)
            overlay = heatmap.overlay(np.zeros((100, 100, 3), dtype=np.uint8))
            self.assertTrue(density_path.is_file())
            self.assertTrue(movement_path.is_file())
            self.assertEqual(overlay.shape, (100, 100, 3))

    def test_rejects_non_normalized_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.json"
            path.write_text(json.dumps({"line_start": [-1.0, 0.5]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                AppConfig.load(path)


class VideoSourceTests(unittest.TestCase):
    def test_file_timestamps_follow_media_position(self) -> None:
        source = VideoSource("clip.mp4", reconnect_attempts=0, reconnect_delay=0)
        source.capture = Mock()
        source.capture.get.side_effect = [0.0, 400.0]
        started_at = datetime(2025, 1, 1, tzinfo=timezone.utc)

        first = source.timestamp(1, 25.0, started_at)
        second = source.timestamp(2, 25.0, started_at)

        self.assertEqual(first, started_at)
        self.assertEqual(second, started_at + timedelta(milliseconds=400))


if __name__ == "__main__":
    unittest.main()
