# YOLOv8 CCTV Video Analytics

Advanced CCTV analytics built with YOLOv8 and ByteTrack for real-time object
detection, tracking, line-crossing analytics, zone monitoring, heatmap
generation, event logging, PDF reporting, SQLite storage, and an interactive
Streamlit dashboard.

A modular video analytics application built on Ultralytics YOLOv8, its ByteTrack
tracker, OpenCV, SQLite, and Streamlit. It processes saved videos, webcams, and
RTSP/IP camera URLs; writes annotated video and event artifacts; and provides a
dashboard and PDF report generator.

## Project layout

```text
YOLOv8-cctv-system/
├── app.py                 # Processing pipeline and command-line interface
├── config.py              # JSON/environment configuration
├── detector.py            # YOLO inference and device selection
├── tracker.py             # ByteTrack adapter
├── counter.py             # Per-frame and unique track counts
├── line_counter.py        # IN/OUT crossings and occupancy
├── zone_analytics.py      # Polygon zone traffic and dwell time
├── database.py            # SQLite schema, event queries, CSV export
├── heatmap.py             # Detection-density image and video overlay
├── video_writer.py        # Checked MP4 writer
├── logger.py              # Console and rotating-file logging
├── dashboard.py           # Streamlit dashboard
├── report_generator.py    # PDF report
├── video_detection.py     # Legacy command entry point
├── config.example.json    # Example settings
├── requirements.txt
├── models/                # Optional location for downloaded model weights
├── data/                  # SQLite database
├── output/                # Video, heatmap, CSV, and logs
└── reports/               # Optional report destination
```

## Installation (Windows PowerShell)

```powershell
cd C:\path\to\YOLOv8-cctv-system
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The included `yolov8n.pt` is used by default. To use a different model, set
`model_path` in `config.json` or `ANALYTICS_MODEL`. CUDA inference is selected
automatically when PyTorch has CUDA support; use `"device": "cpu"` to force CPU.
The Ultralytics package's installed PyTorch build must match the target machine's
CUDA setup.

## Run a video or live camera

```powershell
python app.py --source marathon.mp4
python app.py --source 0 --show
python app.py --source "rtsp://username:password@camera-address/stream"
```

For a dashboard, process a source first, then run:

```powershell
streamlit run dashboard.py
```

Create a PDF from the current database without processing another video:

```powershell
python app.py --report-only
```

`python video_detection.py` remains available as a compatibility entry point.
Set `ANALYTICS_SOURCE` to override the default `marathon.mp4`.

## Configuration

Copy `config.example.json` to `config.json`, then adjust paths, thresholds,
virtual-line coordinates, and zones. Coordinates for lines and polygon vertices
are normalized fractions of the frame (`0.0` through `1.0`), so the same
configuration scales with camera resolution. The default horizontal line
counts travel from below to above as IN and above to below as OUT.

Supported environment overrides include `ANALYTICS_MODEL`,
`ANALYTICS_SOURCE`, `ANALYTICS_OUTPUT_DIR`, `ANALYTICS_DATABASE`,
`ANALYTICS_CONFIDENCE`, `ANALYTICS_DEVICE`, `ANALYTICS_LOG_LEVEL`, and
`ANALYTICS_LOG_FILE`. Keep camera credentials in a protected environment or
secret store, never in a committed configuration file.

## Analytics and outputs

ByteTrack IDs feed the unique-track counter, virtual line, and polygon-zone
analytics. A crossing is counted only when an ID moves from one side of the
configured line to the other; a small deadband suppresses jitter near the line.
Occupancy is the non-negative cumulative `IN - OUT` estimate. Zone traffic
counts entries, and dwell time is measured between zone enter/exit events.
CSV/PDF event exports are available alongside:

* `output/processed_video.mp4` — boxes, IDs, frame/unique counts, line, and occupancy.
* `output/heatmap.png` — accumulated detection-center density.
* `output/movement_heatmap.png` — accumulated tracked-object movement paths.
* `output/heatmap_overlay.mp4` — the running density and movement overlay on annotated video.
* `output/events.csv` — chronological event export.
* `output/analytics_report.pdf` — executive summary, counts, distribution, heatmap, and recommendation.
* `data/analytics.db` — persistent SQLite event database.

Event CSV exports include the processing `session_id`. Event timestamps are UTC;
for prerecorded videos, timestamp intervals follow the media position (anchored
to processing start), while live sources use wall-clock time. If a run has no
detections, prior heatmap images are removed so they are not mistaken for the
current run's results.

SQLite schema:

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    object_id INTEGER,
    object_class TEXT NOT NULL,
    event_type TEXT NOT NULL,
    direction TEXT,
    frame_number INTEGER NOT NULL
);
```

Event types include `TRACK_STARTED`, `LINE_CROSSING_IN`,
`LINE_CROSSING_OUT`, `ZONE_ENTER`, and `ZONE_EXIT`. Run independent cameras
with separate database/output paths to prevent tracker-state and artifact
collisions. Each processing run has a session record so track IDs and occupancy
are summarized within the selected run, while prior events remain available.

## Implementation steps

1. Install dependencies and verify the model-weight path in configuration.
2. Set the source and draw the virtual line position for the target camera.
3. Add normalized polygon vertices for any monitored zones.
4. Run `python app.py --source <file-or-camera>` and review the processed video,
   database, and CSV export.
5. Start `streamlit run dashboard.py` to inspect events and traffic charts.
6. Generate or download the PDF report from the dashboard or CLI.

## Deployment notes

ByteTrack provides motion-based association, not appearance-based
re-identification. IDs can change after occlusion, reconnect, or tracker reset;
unique counts are unique IDs within a processing run, not guaranteed identities
across cameras or sessions. Occupancy is meaningful only when the camera and
line cover every entry/exit path. Configure camera reconnect limits, storage
retention, authentication, network isolation, and privacy controls for the
deployment. For high availability or many cameras, run isolated workers and
move event persistence to a managed database/message queue rather than
sharing a single local SQLite file. Validate throughput, line direction,
camera-specific zone placement, and alerting with representative footage
before production use.
