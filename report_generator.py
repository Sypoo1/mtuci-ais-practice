from datetime import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER


def generate_pdf_report(session_data: list, session_id: str) -> BytesIO:
    """
    Generate PDF report for user session analytics.

    Args:
        session_data: List of analytics records
        session_id: User session ID

    Returns:
        BytesIO: PDF file in memory
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#111F68'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    title = Paragraph("MTUCI Shop Detector<br/>Analytics Report", title_style)
    elements.append(title)

    info_style = styles['Normal']
    info_text = f"<b>Session ID:</b> {session_id}<br/><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/><b>Total Records:</b> {len(session_data)}"
    info = Paragraph(info_text, info_style)
    elements.append(info)
    elements.append(Spacer(1, 0.3*inch))

    if not session_data:
        no_data = Paragraph("<b>No analytics data found for this session.</b>", styles['Normal'])
        elements.append(no_data)
    else:
        total_images = sum(1 for r in session_data if r['file_type'] == 'image')
        total_videos = sum(1 for r in session_data if r['file_type'] == 'video')

        summary_text = f"<b>Summary:</b><br/>Images processed: {total_images}<br/>Videos processed: {total_videos}"
        summary = Paragraph(summary_text, styles['Normal'])
        elements.append(summary)
        elements.append(Spacer(1, 0.3*inch))

        table_title = Paragraph("<b>Detection Records</b>", styles['Heading2'])
        elements.append(table_title)
        elements.append(Spacer(1, 0.2*inch))

        table_data = [['#', 'File', 'Type', 'Count', 'Min', 'Max', 'Avg', 'Conf', 'IoU', 'Model']]

        for idx, record in enumerate(session_data, 1):
            conf_val = f"{record['confidence_threshold']:.2f}" if record['confidence_threshold'] else '-'
            iou_val = f"{record['iou_threshold']:.2f}" if record['iou_threshold'] else '-'

            if record['file_type'] == 'image':
                row = [
                    str(idx),
                    record['file_name'][:15] + '...' if len(record['file_name']) > 15 else record['file_name'],
                    record['file_type'],
                    str(record['person_count'] or '-'),
                    '-',
                    '-',
                    '-',
                    conf_val,
                    iou_val,
                    record['model_name'][:8] if record['model_name'] else '-'
                ]
            else:
                row = [
                    str(idx),
                    record['file_name'][:15] + '...' if len(record['file_name']) > 15 else record['file_name'],
                    record['file_type'],
                    '-',
                    str(record['person_count_min']) if record['person_count_min'] is not None else '-',
                    str(record['person_count_max']) if record['person_count_max'] is not None else '-',
                    f"{record['person_count_avg']:.1f}" if record['person_count_avg'] is not None else '-',
                    conf_val,
                    iou_val,
                    record['model_name'][:8] if record['model_name'] else '-'
                ]
            table_data.append(row)

        table = Table(table_data, colWidths=[0.4*inch, 1.3*inch, 0.5*inch, 0.5*inch, 0.5*inch, 0.5*inch, 0.5*inch, 0.5*inch, 0.5*inch, 0.8*inch])

        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#111F68')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))

        elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer