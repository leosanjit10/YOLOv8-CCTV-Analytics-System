"""Video analytics pipeline and command-line entry point."""

from __future__ import annotations

import argparse
from math import isfinite
import time
from datetime import datetime, timedelta, timezone

import cv2

from config import AppConfig
from counter import ObjectCounter
from database import AnalyticsDatabase
from detector import Detection, Detector
from heatmap import HeatmapAnalytics
from line_counter import LineCounter
from logger import get_logger
from report_generator import generate_report
from tracker import ByteTrackTracker
from video_writer import VideoWriter
from zone_analytics import ZoneAnalytics


class VideoSource:
    def __init__(self, source: str, reconnect_attempts: int, reconnect_delay: float) -> None:
        self.source = self._parse_source(source)
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.capture: cv2.VideoCapture | None = None
        self._last_media_seconds: float | None = None
        self.is_live = isinstance(self.source, int) or (
            isinstance(self.source, str)
            and self.source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
        )

    @staticmethod
    def _parse_source(source: str) -> str | int:
        return int(source) if source.isdigit() else source

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            self.capture.release()
            raise OSError(f"Could not open video source: {self.source}")

    def read(self):
        if self.capture is None:
            raise RuntimeError("Video source has not been opened")
        failures = 0
        while True:
            ok, frame = self.capture.read()
            if ok:
                return frame
            if not self.is_live:
                return None
            failures += 1
            if failures > self.reconnect_attempts:
                raise OSError(
                    f"Live stream {self.source!r} failed after "
                    f"{self.reconnect_attempts} reconnect attempts"
                )
            self.capture.release()
            time.sleep(self.reconnect_delay)
            self.capture = cv2.VideoCapture(self.source)

    def timestamp(
        self, frame_number: int, fps: float, started_at: datetime
    ) -> datetime:
        if self.is_live:
            return datetime.now(timezone.utc)
        if self.capture is None:
            raise RuntimeError("Video source has not been opened")

        safe_fps = fps if isfinite(fps) and fps > 0 else 25.0
        fallback_seconds = (frame_number - 1) / safe_fps
        position_msec = self.capture.get(cv2.CAP_PROP_POS_MSEC)
        reported_seconds = position_msec / 1000
        if (
            isfinite(position_msec)
            and position_msec >= 0
            and (
                self._last_media_seconds is None
                or reported_seconds > self._last_media_seconds
            )
        ):
            media_seconds = reported_seconds
        else:
            media_seconds = fallback_seconds
        if (
            self._last_media_seconds is not None
            and media_seconds <= self._last_media_seconds
        ):
            media_seconds = self._last_media_seconds + 1 / safe_fps
        self._last_media_seconds = media_seconds
        return started_at + timedelta(seconds=media_seconds)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class VideoAnalyticsApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = get_logger("video-analytics", config.log_level, config.log_file)
        self.detector = Detector(
            config.model_path,
            confidence=config.confidence,
            iou=config.iou,
            image_size=config.image_size,
            device=config.device,
            classes=config.classes,
        )
        self.tracker = ByteTrackTracker(self.detector)
        self.counter = ObjectCounter()
        self.line_counter = LineCounter(config.line_start, config.line_end)
        self.zones = ZoneAnalytics(config.zones)
        self.heatmap = HeatmapAnalytics()
        self.database = AnalyticsDatabase(config.database_path)
        self._logged_ids: set[int] = set()

    def run(self, source: str | None = None, show: bool = False) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        video_source = VideoSource(
            source or self.config.source,
            self.config.reconnect_attempts,
            self.config.reconnect_delay_seconds,
        )
        video_source.open()
        assert video_source.capture is not None
        fps = video_source.capture.get(cv2.CAP_PROP_FPS)
        width = int(video_source.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_source.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            video_source.close()
            raise OSError("Video source reported invalid frame dimensions")
        processed_path = self.config.output_dir / "processed_video.mp4"
        heatmap_video_path = self.config.output_dir / "heatmap_overlay.mp4"
        frame_number = 0
        session_id = self.database.start_session(str(source or self.config.source))
        started_at = datetime.now(timezone.utc)
        try:
            with VideoWriter(processed_path, fps, (width, height)) as processed_writer, \
                    VideoWriter(heatmap_video_path, fps, (width, height)) as heatmap_writer:
                while True:
                    frame = video_source.read()
                    if frame is None:
                        break
                    frame_number += 1
                    timestamp = video_source.timestamp(frame_number, fps, started_at)
                    detections = self.tracker.update(frame)
                    self.counter.update(detections)
                    self.heatmap.update(frame.shape, detections)

                    for detection in detections:
                        if detection.track_id is not None and detection.track_id not in self._logged_ids:
                            self._logged_ids.add(detection.track_id)
                            self.database.log_event(
                                detection, "TRACK_STARTED", frame_number,
                                timestamp=timestamp, session_id=session_id,
                            )

                    for detection, direction in self.line_counter.update(detections, frame.shape):
                        self.database.log_event(
                            detection, f"LINE_CROSSING_{direction}", frame_number,
                            direction=direction, timestamp=timestamp,
                            session_id=session_id,
                        )
                    for detection, zone, event_type, duration in self.zones.update(
                        detections, frame.shape, timestamp
                    ):
                        self.database.log_event(
                            detection, event_type, frame_number,
                            direction=f"{zone}:{duration:.2f}s" if event_type == "ZONE_EXIT" else zone,
                            timestamp=timestamp, session_id=session_id,
                        )

                    annotated = self._annotate(frame.copy(), detections, frame_number)
                    processed_writer.write(annotated)
                    heatmap_writer.write(self.heatmap.overlay(annotated))
                    if show:
                        cv2.imshow("Video Analytics (press Q to stop)", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        finally:
            video_source.close()
            if show:
                cv2.destroyAllWindows()
            self.database.finish_session(
                session_id,
                self.counter.total_detections,
                self.counter.unique_count,
                self.line_counter.entries,
                self.line_counter.exits,
                self.line_counter.occupancy,
            )

        heatmap_path = self.config.output_dir / "heatmap.png"
        movement_heatmap_path = self.config.output_dir / "movement_heatmap.png"
        if self.counter.total_detections:
            self.heatmap.save(heatmap_path)
            self.heatmap.save_movement(movement_heatmap_path)
        else:
            heatmap_path.unlink(missing_ok=True)
            movement_heatmap_path.unlink(missing_ok=True)
        self.database.export_csv(self.config.output_dir / "events.csv")
        report_path = generate_report(
            self.database, self.config.output_dir / "analytics_report.pdf", heatmap_path
        )
        self.logger.info(
            "Processed %d frames; %d unique tracked objects; entries=%d exits=%d occupancy=%d",
            frame_number, self.counter.unique_count, self.line_counter.entries,
            self.line_counter.exits, self.line_counter.occupancy,
        )
        self.logger.info("Artifacts: %s, %s", processed_path, report_path)

    def _annotate(
        self, frame, detections: list[Detection], frame_number: int
    ):
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            color = (0, 210, 80)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{detection.class_name} #{detection.track_id} {detection.confidence:.2f}"
            cv2.putText(
                frame, label, (x1, max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )
            cv2.circle(frame, detection.center, 3, (0, 0, 255), -1)
        height, width = frame.shape[:2]
        start = LineCounter._to_pixels(self.config.line_start, width, height)
        end = LineCounter._to_pixels(self.config.line_end, width, height)
        cv2.line(frame, start, end, (255, 170, 0), 2)
        overlay = (
            f"Frame {frame_number} | Current {len(detections)} | "
            f"Unique {self.counter.unique_count} | "
            f"IN {self.line_counter.entries} OUT {self.line_counter.exits} "
            f"Occupancy {self.line_counter.occupancy}"
        )
        cv2.putText(frame, overlay, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        class_counts = "  ".join(
            f"{name}: {count}"
            for name, count in self.counter.unique_by_class.most_common(5)
        )
        if class_counts:
            cv2.putText(
                frame, class_counts, (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )
        return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 CCTV video analytics")
    parser.add_argument("--source", help="Video path, webcam index, or RTSP/IP camera URL")
    parser.add_argument("--config", help="Optional JSON configuration file")
    parser.add_argument("--show", action="store_true", help="Display the annotated stream")
    parser.add_argument("--report-only", action="store_true", help="Generate PDF from the existing database")
    args = parser.parse_args()

    config = AppConfig.load(args.config)
    if args.report_only:
        database = AnalyticsDatabase(config.database_path)
        path = generate_report(
            database, config.output_dir / "analytics_report.pdf",
            config.output_dir / "heatmap.png",
        )
        print(f"Report saved to {path}")
        return
    VideoAnalyticsApp(config).run(source=args.source, show=args.show)


if __name__ == "__main__":
    main()
