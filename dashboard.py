"""Streamlit analytics dashboard. Launch with: streamlit run dashboard.py"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config import AppConfig
from database import AnalyticsDatabase
from report_generator import generate_report


def database_csv_bytes(
    database: AnalyticsDatabase, session_id: str | None = None
) -> bytes:
    import csv
    import io

    output = io.StringIO()
    rows = database.fetch_events(session_id=session_id)
    fieldnames = (
        ["id", "session_id", "timestamp", "object_id", "object_class",
         "event_type", "direction", "frame_number"]
    )
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(dict(row) for row in reversed(rows))
    return output.getvalue().encode("utf-8")


st.set_page_config(page_title="CCTV Video Analytics", page_icon="📹", layout="wide")
config = AppConfig.load()
database = AnalyticsDatabase(config.database_path)
sessions = database.list_sessions()
session_labels = {
    row["session_id"]: f"{row['source']} — {row['started_at']}"
    for row in sessions
}
selected_session = (
    st.selectbox(
        "Processing session",
        options=list(session_labels),
        format_func=lambda session_id: session_labels[session_id],
    )
    if session_labels
    else None
)
events = pd.DataFrame(
    [
        dict(row)
        for row in database.fetch_events(limit=5000, session_id=selected_session)
    ]
)
summary = database.summary(selected_session)

st.title("CCTV Video Analytics")
st.caption(f"Event database: {config.database_path}")
if events.empty:
    st.info("No analytics events yet. Process a video or connect a camera with `python app.py --source 0`.")
else:
    track_events = events[events["event_type"] == "TRACK_STARTED"]
    total_objects = int(track_events["object_id"].nunique())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Unique tracked objects", total_objects)
    col2.metric("Estimated occupancy", summary["occupancy"])
    col3.metric("Entries", summary["entries"])
    col4.metric("Exits", summary["exits"])

    st.subheader("Per-class unique tracks")
    class_data = pd.DataFrame(
        [{"Class": name, "Unique tracks": count}
         for name, count in summary["class_counts"].items()]
    )
    if not class_data.empty:
        st.plotly_chart(
            px.bar(class_data, x="Class", y="Unique tracks", color="Class"),
            use_container_width=True,
        )

    zone_entries = events[events["event_type"] == "ZONE_ENTER"]
    if not zone_entries.empty:
        st.subheader("Zone traffic")
        zone_data = (
            zone_entries["direction"]
            .value_counts()
            .rename_axis("Zone")
            .reset_index(name="Visits")
        )
        zone_exits = events[events["event_type"] == "ZONE_EXIT"].copy()
        if not zone_exits.empty:
            zone_exits["Zone"] = zone_exits["direction"].str.rsplit(":", n=1).str[0]
            zone_exits["Dwell seconds"] = pd.to_numeric(
                zone_exits["direction"].str.extract(r":([0-9.]+)s$")[0],
                errors="coerce",
            )
            dwell = zone_exits.groupby("Zone")["Dwell seconds"].mean()
            zone_data["Average dwell (s)"] = zone_data["Zone"].map(dwell)
        st.dataframe(zone_data, use_container_width=True, hide_index=True)
        st.metric(
            "Most visited zone",
            zone_data.sort_values("Visits", ascending=False).iloc[0]["Zone"],
        )

    crossings = events[events["event_type"].str.startswith("LINE_CROSSING")].copy()
    if not crossings.empty:
        crossings["timestamp"] = pd.to_datetime(crossings["timestamp"], utc=True)
        crossings["hour"] = crossings["timestamp"].dt.floor("h")
        hourly = crossings.groupby(["hour", "direction"]).size().reset_index(name="Count")
        left, right = st.columns(2)
        left.plotly_chart(
            px.line(hourly, x="hour", y="Count", color="direction", title="Hourly traffic"),
            use_container_width=True,
        )
        totals = crossings.groupby("direction").size().reset_index(name="Count")
        right.plotly_chart(
            px.bar(totals, x="direction", y="Count", title="Entry / exit graph"),
            use_container_width=True,
        )

    st.subheader("Detection density heatmap")
    heatmap_path = config.output_dir / "heatmap.png"
    if heatmap_path.is_file():
        st.image(str(heatmap_path), use_container_width=True)
    else:
        st.caption("Process a video to generate a heatmap.")

    st.subheader("Recent event log and detections")
    st.dataframe(events.head(100), use_container_width=True, hide_index=True)
    csv_bytes = database_csv_bytes(database, selected_session)
    st.download_button("Download events CSV", csv_bytes, "events.csv", "text/csv")

    if st.button("Generate PDF report"):
        report_path = generate_report(
            database, config.output_dir / "analytics_report.pdf", heatmap_path,
            session_id=selected_session,
        )
        st.success(f"Report saved: {report_path}")
        st.download_button(
            "Download PDF report", report_path.read_bytes(), report_path.name,
            "application/pdf",
        )
