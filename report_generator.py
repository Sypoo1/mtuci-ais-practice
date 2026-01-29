from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

TITLE_COLOR = "#111F68"
TABLE_HEADER = [
    "#",
    "File",
    "Type",
    "Count",
    "Min",
    "Max",
    "Avg",
    "Conf",
    "IoU",
    "Model",
]
COL_WIDTHS = [0.4, 1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8]


def generate_pdf_report(session_data: list, session_id: str) -> BytesIO:
    """Generate PDF report for user session analytics."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = [
        _create_title(styles),
        _create_info_section(session_id, len(session_data), styles),
        Spacer(1, 0.3 * inch),
    ]

    if not session_data:
        elements.append(
            Paragraph(
                "<b>No analytics data found for this session.</b>", styles["Normal"]
            )
        )
    else:
        elements.extend(_create_data_section(session_data, styles))

    doc.build(elements)
    buffer.seek(0)
    return buffer


def _create_title(styles) -> Paragraph:
    """Create report title."""
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor(TITLE_COLOR),
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    return Paragraph("MTUCI Shop Detector<br/>Analytics Report", title_style)


def _create_info_section(session_id: str, record_count: int, styles) -> Paragraph:
    """Create session information section."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_text = (
        f"<b>Session ID:</b> {session_id}<br/>"
        f"<b>Generated:</b> {timestamp}<br/>"
        f"<b>Total Records:</b> {record_count}"
    )
    return Paragraph(info_text, styles["Normal"])


def _create_data_section(session_data: list, styles) -> list:
    """Create data section with summary and table."""
    total_images = sum(1 for r in session_data if r["file_type"] == "image")
    total_videos = sum(1 for r in session_data if r["file_type"] == "video")

    summary_text = (
        f"<b>Summary:</b><br/>"
        f"Images processed: {total_images}<br/>"
        f"Videos processed: {total_videos}"
    )

    return [
        Paragraph(summary_text, styles["Normal"]),
        Spacer(1, 0.3 * inch),
        Paragraph("<b>Detection Records</b>", styles["Heading2"]),
        Spacer(1, 0.2 * inch),
        _create_table(session_data),
    ]


def _create_table(session_data: list) -> Table:
    """Create data table."""
    table_data = [TABLE_HEADER]
    table_data.extend(
        _create_table_row(idx, record) for idx, record in enumerate(session_data, 1)
    )

    table = Table(table_data, colWidths=[w * inch for w in COL_WIDTHS])
    table.setStyle(_get_table_style())
    return table


def _create_table_row(idx: int, record: dict) -> list:
    """Create a single table row."""
    conf_val = (
        f"{record['confidence_threshold']:.2f}"
        if record["confidence_threshold"]
        else "-"
    )
    iou_val = f"{record['iou_threshold']:.2f}" if record["iou_threshold"] else "-"
    file_name = _truncate_text(record["file_name"], 15)
    model_name = (
        _truncate_text(record["model_name"], 8) if record["model_name"] else "-"
    )

    if record["file_type"] == "image":
        return [
            str(idx),
            file_name,
            record["file_type"],
            str(record["person_count"] or "-"),
            "-",
            "-",
            "-",
            conf_val,
            iou_val,
            model_name,
        ]

    return [
        str(idx),
        file_name,
        record["file_type"],
        "-",
        str(record["person_count_min"])
        if record["person_count_min"] is not None
        else "-",
        str(record["person_count_max"])
        if record["person_count_max"] is not None
        else "-",
        f"{record['person_count_avg']:.1f}"
        if record["person_count_avg"] is not None
        else "-",
        conf_val,
        iou_val,
        model_name,
    ]


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max length with ellipsis."""
    return text[:max_length] + "..." if len(text) > max_length else text


def _get_table_style() -> TableStyle:
    """Get table styling."""
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TITLE_COLOR)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]
    )
