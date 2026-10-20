# YOLOv8 CCTV Video Analytics System

An end-to-end CCTV analytics platform built with **YOLOv8**, **ByteTrack**, **OpenCV**, **SQLite**, and **Streamlit**.

The system performs real-time object detection, multi-object tracking, line-crossing analytics, zone monitoring, heatmap generation, event logging, automated PDF reporting, and interactive analytics visualization.

---

## 🚀 Key Features

### Computer Vision

- YOLOv8 Object Detection
- ByteTrack Multi-Object Tracking
- Real-Time Video Analytics
- Persistent Object Tracking IDs
- Class-Based Detection Statistics

### Analytics

- Line Crossing Detection
- Entry / Exit Counting
- Occupancy Estimation
- Zone-Based Monitoring
- Dwell Time Analysis
- Detection Density Heatmaps
- Movement Heatmaps
- Traffic Flow Analysis

### Data & Reporting

- SQLite Event Storage
- Session Tracking
- Historical Analytics
- CSV Export
- Automated PDF Reports
- Event Logging

### Dashboard

- Interactive Streamlit Dashboard
- Session Metrics
- Analytics Charts
- Event Exploration
- Heatmap Visualization
- CSV Downloads
- PDF Report Access

---

## 📸 Dashboard Preview

> Add screenshots here after uploading them to the repository.

### Dashboard

![Dashboard](screenshots/dashboard.png)

### Heatmap

![Heatmap](screenshots/heatmap.png)

### Processed Video

![Processed Video](screenshots/processed_video.png)

---

## 🏗️ System Architecture

```text
Video Input
    │
    ▼
YOLOv8 Detection
    │
    ▼
ByteTrack Tracking
    │
    ▼
Analytics Engine
 ┌──────────────────┐
 │ Object Counting  │
 │ Line Crossing    │
 │ Zone Analytics   │
 │ Occupancy        │
 │ Dwell Time       │
 │ Heatmaps         │
 └──────────────────┘
    │
    ▼
SQLite Database
    │
    ├── CSV Export
    ├── PDF Report
    └── Streamlit Dashboard
```

---

## 📂 Project Structure

```text
YOLOv8-cctv-system/

├── app.py                 # Main processing pipeline
├── video_detection.py     # Legacy entry point
├── config.py             # Configuration management
├── detector.py           # YOLOv8 inference
├── tracker.py            # ByteTrack adapter
├── counter.py            # Unique track counting
├── line_counter.py       # Line-crossing analytics
├── zone_analytics.py     # Zone monitoring & dwell analytics
├── heatmap.py            # Heatmap generation
├── video_writer.py       # Processed video generation
├── database.py           # SQLite storage & queries
├── logger.py             # Logging utilities
├── report_generator.py   # PDF reporting
├── dashboard.py          # Streamlit dashboard

├── tests/
│   └── test_analytics.py

├── output/
├── data/

├── config.example.json
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

### AI & Computer Vision

- YOLOv8
- ByteTrack
- OpenCV

### Backend

- Python
- SQLite
- Pandas

### Visualization

- Streamlit
- Matplotlib

### Reporting

- ReportLab

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/leosanjit10/YOLOv8-CCTV-Analytics-System.git

cd YOLOv8-CCTV-Analytics-System
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎥 Run Video Analytics

Process a local video:

```bash
python app.py --source marathon.mp4
```

Process webcam:

```bash
python app.py --source 0 --show
```

Process RTSP camera:

```bash
python app.py --source "rtsp://username:password@camera-address/stream"
```

Compatibility entry point:

```bash
python video_detection.py
```

Generate a PDF report from existing analytics:

```bash
python app.py --report-only
```

---

## 📊 Launch Dashboard

```bash
streamlit run dashboard.py
```

Then open:

```text
http://localhost:8501
```

The dashboard provides:

- Unique Track Statistics
- Entry / Exit Analytics
- Occupancy Metrics
- Traffic Analytics
- Heatmaps
- Event Logs
- CSV Downloads
- PDF Report Access

---

## 📈 Generated Outputs

After processing a video, the system generates:

```text
output/

├── processed_video.mp4
├── heatmap_overlay.mp4
├── heatmap.png
├── movement_heatmap.png
├── events.csv
└── analytics_report.pdf

data/

└── analytics.db
```

### Processed Video

Annotated output containing:

- Bounding Boxes
- Class Labels
- Track IDs
- Entry / Exit Counts
- Occupancy Metrics

### Heatmaps

- Detection Density Heatmap
- Movement Activity Heatmap

### PDF Report

Automatically generated report containing:

- Traffic Summary
- Object Statistics
- Occupancy Metrics
- Visual Analytics
- Heatmap Visualizations

---

## 📊 Analytics Capabilities

### Object Analytics

- Unique Object Counting
- Class-wise Distribution
- Detection Statistics

### Line-Crossing Analytics

- Entry Detection
- Exit Detection
- Occupancy Estimation
- Traffic Flow Monitoring

### Zone Analytics

- Zone Entry Detection
- Zone Exit Detection
- Dwell Time Calculation
- Zone Occupancy Analysis

### Heatmap Analytics

- Detection Density Mapping
- Movement Flow Visualization
- Hotspot Identification

### Database & Reporting

- Persistent Event Storage
- Session-Based Analytics
- CSV Export
- PDF Export
- Historical Analysis

---

## 🧪 Testing

Run the analytics test suite:

```bash
python -m unittest discover -s tests -v
```

Covered functionality:

- Line Crossing Logic
- Zone Analytics
- Heatmap Generation
- Database Operations
- CSV Export Validation
- Configuration Validation

---

## 📋 Typical Workflow

### Step 1

Process video:

```bash
python app.py --source marathon.mp4
```

### Step 2

Generated artifacts:

```text
processed_video.mp4
events.csv
analytics.db
analytics_report.pdf
heatmap.png
movement_heatmap.png
```

### Step 3

Launch dashboard:

```bash
streamlit run dashboard.py
```

### Step 4

Analyze results interactively through the dashboard.

---

## 🔮 Future Enhancements

- Video Upload Through Dashboard
- Real-Time Webcam Analytics
- RTSP Multi-Camera Monitoring
- Live Stream Processing
- Intrusion Detection
- Loitering Detection
- Email & SMS Alerts
- User Authentication
- REST API
- Docker Deployment
- Cloud Deployment
- Distributed Multi-Camera Architecture

---

## 💼 Portfolio Highlights

This project demonstrates:

- Computer Vision
- Object Detection
- Multi-Object Tracking
- Video Analytics
- Data Engineering
- Dashboard Development
- Report Generation
- Database Design
- Software Architecture
- End-to-End AI System Development

---

## 👨‍💻 Author

**Sanjit Sitaula**

GitHub: https://github.com/leosanjit10

---

## 📄 License

This project is provided for educational, research, and portfolio purposes.
