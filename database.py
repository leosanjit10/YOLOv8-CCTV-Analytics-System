"""SQLite event persistence and CSV export."""

from __future__ import annotations

import csv
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from detector import Detection


class AnalyticsDatabase:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT 'legacy',
                    timestamp TEXT NOT NULL,
                    object_id INTEGER,
                    object_class TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    direction TEXT,
                    frame_number INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    total_detections INTEGER NOT NULL DEFAULT 0,
                    unique_objects INTEGER NOT NULL DEFAULT 0,
                    entries INTEGER NOT NULL DEFAULT 0,
                    exits INTEGER NOT NULL DEFAULT 0,
                    occupancy INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            event_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(events)")
            }
            if "session_id" not in event_columns:
                connection.execute(
                    "ALTER TABLE events ADD COLUMN session_id TEXT NOT NULL DEFAULT 'legacy'"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)"
            )

    def start_session(self, source: str) -> str:
        session_id = str(uuid4())
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO sessions(session_id, source, started_at) VALUES (?, ?, ?)",
                (session_id, source, datetime.now(timezone.utc).isoformat()),
            )
        return session_id

    def finish_session(
        self,
        session_id: str,
        total_detections: int,
        unique_objects: int,
        entries: int,
        exits: int,
        occupancy: int,
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE sessions SET ended_at = ?, total_detections = ?,
                    unique_objects = ?, entries = ?, exits = ?, occupancy = ?
                WHERE session_id = ?
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    total_detections,
                    unique_objects,
                    entries,
                    exits,
                    occupancy,
                    session_id,
                ),
            )

    def list_sessions(self) -> list[sqlite3.Row]:
        with closing(self._connect()) as connection, connection:
            return list(
                connection.execute(
                    "SELECT * FROM sessions ORDER BY started_at DESC"
                )
            )

    def log_event(
        self,
        detection: Detection,
        event_type: str,
        frame_number: int,
        direction: str | None = None,
        timestamp: datetime | None = None,
        session_id: str = "legacy",
    ) -> None:
        event_time = timestamp or datetime.now(timezone.utc)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO events(session_id, timestamp, object_id, object_class,
                                   event_type, direction, frame_number)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    event_time.astimezone(timezone.utc).isoformat(),
                    detection.track_id,
                    detection.class_name,
                    event_type,
                    direction,
                    frame_number,
                ),
            )

    def fetch_events(
        self, limit: int | None = None, session_id: str | None = None
    ) -> list[sqlite3.Row]:
        query = "SELECT * FROM events"
        parameters: list[object] = []
        if session_id is not None:
            query += " WHERE session_id = ?"
            parameters.append(session_id)
        query += " ORDER BY timestamp DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with closing(self._connect()) as connection, connection:
            return list(connection.execute(query, parameters))

    def export_csv(self, csv_path: Path) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.fetch_events()
        fieldnames = (
            ["id", "session_id", "timestamp", "object_id", "object_class",
             "event_type", "direction", "frame_number"]
        )
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dict(row) for row in reversed(rows))

    def summary(self, session_id: str | None = None) -> dict[str, object]:
        if session_id is None:
            sessions = self.list_sessions()
            session_id = sessions[0]["session_id"] if sessions else "legacy"
        with closing(self._connect()) as connection, connection:
            event_counts = {
                row["event_type"]: row["count"]
                for row in connection.execute(
                    """SELECT event_type, COUNT(*) AS count FROM events
                       WHERE session_id = ? GROUP BY event_type""",
                    (session_id,),
                )
            }
            class_counts = {
                row["object_class"]: row["count"]
                for row in connection.execute(
                    """SELECT object_class, COUNT(DISTINCT object_id) AS count
                       FROM events WHERE object_id IS NOT NULL AND session_id = ?
                       GROUP BY object_class ORDER BY count DESC""",
                    (session_id,),
                )
            }
            hourly = [
                dict(row)
                for row in connection.execute(
                    """SELECT substr(timestamp, 1, 13) AS hour, COUNT(*) AS count
                       FROM events WHERE event_type LIKE 'LINE_CROSSING_%'
                         AND session_id = ?
                       GROUP BY hour ORDER BY hour""",
                    (session_id,),
                )
            ]
        entries = event_counts.get("LINE_CROSSING_IN", 0)
        exits = event_counts.get("LINE_CROSSING_OUT", 0)
        return {
            "session_id": session_id,
            "entries": entries,
            "exits": exits,
            "occupancy": max(0, entries - exits),
            "event_counts": event_counts,
            "class_counts": class_counts,
            "hourly": hourly,
        }
