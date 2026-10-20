"""Direction-aware virtual-line crossing analytics."""

from collections import Counter

from detector import Detection


class LineCounter:
    def __init__(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        deadband_pixels: float = 4.0,
    ) -> None:
        if start == end:
            raise ValueError("The virtual line must have two different endpoints")
        self.start = start
        self.end = end
        self.deadband_pixels = deadband_pixels
        self._last_side: dict[int, int] = {}
        self.entries = 0
        self.exits = 0
        self.by_class: Counter[str] = Counter()

    def update(
        self, detections: list[Detection], frame_shape: tuple[int, ...]
    ) -> list[tuple[Detection, str]]:
        height, width = frame_shape[:2]
        x1, y1 = self._to_pixels(self.start, width, height)
        x2, y2 = self._to_pixels(self.end, width, height)
        crossings: list[tuple[Detection, str]] = []
        for detection in detections:
            if detection.track_id is None:
                continue
            x, y = detection.center
            signed_area = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            length = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1.0)
            distance = signed_area / length
            if abs(distance) <= self.deadband_pixels:
                continue
            side = 1 if distance > 0 else -1
            previous_side = self._last_side.get(detection.track_id)
            self._last_side[detection.track_id] = side
            if previous_side is None or previous_side == side:
                continue

            direction = "IN" if previous_side > side else "OUT"
            if direction == "IN":
                self.entries += 1
            else:
                self.exits += 1
            self.by_class[detection.class_name] += 1
            crossings.append((detection, direction))
        return crossings

    @staticmethod
    def _to_pixels(
        point: tuple[float, float], width: int, height: int
    ) -> tuple[int, int]:
        return (int(point[0] * width), int(point[1] * height))

    @property
    def occupancy(self) -> int:
        return max(0, self.entries - self.exits)
