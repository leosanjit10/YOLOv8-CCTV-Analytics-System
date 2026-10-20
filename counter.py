"""Per-frame and per-track object counting."""

from collections import Counter

from detector import Detection


class ObjectCounter:
    def __init__(self) -> None:
        self.total_detections = 0
        self.unique_ids: set[int] = set()
        self.unique_by_class: Counter[str] = Counter()
        self.detections_by_class: Counter[str] = Counter()

    def update(self, detections: list[Detection]) -> None:
        self.total_detections += len(detections)
        for detection in detections:
            self.detections_by_class[detection.class_name] += 1
            if detection.track_id is not None and detection.track_id not in self.unique_ids:
                self.unique_ids.add(detection.track_id)
                self.unique_by_class[detection.class_name] += 1

    @property
    def unique_count(self) -> int:
        return len(self.unique_ids)
