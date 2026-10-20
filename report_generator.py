"""Generate a printable analytics summary PDF."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from database import AnalyticsDatabase


def generate_report(
    database: AnalyticsDatabase,
    output_path: Path,
    heatmap_path: Path | None = None,
    session_id: str | None = None,
) -> Path:
    summary = database.summary(session_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Video Analytics Report", styles["Title"]),
        Paragraph(
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            styles["Normal"],
        ),
        Spacer(1, 16),
        Paragraph("Executive summary", styles["Heading2"]),
        Paragraph(
            f"Observed {sum(summary['class_counts'].values())} unique tracked objects, "
            f"{summary['entries']} entries and {summary['exits']} exits. "
            f"Estimated current occupancy: {summary['occupancy']}.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        Paragraph("Traffic and object distribution", styles["Heading2"]),
    ]
    rows = [["Object class", "Unique tracked objects"]]
    rows.extend(
        [name, str(count)] for name, count in summary["class_counts"].items()
    )
    rows.extend(
        [["Entries", str(summary["entries"])], ["Exits", str(summary["exits"])],
         ["Current occupancy", str(summary["occupancy"])]]
    )
    table = Table(rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([table, Spacer(1, 16)])
    if heatmap_path is not None and heatmap_path.is_file():
        story.extend(
            [
                Paragraph("Detection density heatmap", styles["Heading2"]),
                Image(str(heatmap_path), width=450, height=260, kind="proportional"),
                Spacer(1, 12),
            ]
        )

    recommendation = (
        "Review peak hourly traffic and staff high-volume periods."
        if summary["entries"] + summary["exits"] > 0
        else "No line-crossing events were recorded. Configure a virtual line and "
             "verify that tracked objects cross it."
    )
    story.extend(
        [
            Paragraph("Recommendations", styles["Heading2"]),
            Paragraph(recommendation, styles["BodyText"]),
            Paragraph(
                "Occupancy is an estimate derived from configured IN/OUT line crossings. "
                "Validate line placement and camera coverage before operational use.",
                styles["BodyText"],
            ),
        ]
    )
    SimpleDocTemplate(str(output_path), pagesize=letter).build(story)
    return output_path
