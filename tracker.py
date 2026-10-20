"""ByteTrack-backed persistent object tracking."""

from detector import Detection, Detector


class ByteTrackTracker:
    """Keep tracking behind a small adapter so analytics do not depend on Ultralytics."""

    def __init__(self, detector: Detector, tracker_config: str = "bytetrack.yaml") -> None:
        self.detector = detector
        self.tracker_config = tracker_config

    def update(self, frame) -> list[Detection]:
        return self.detector.track_frame(frame, self.tracker_config)
