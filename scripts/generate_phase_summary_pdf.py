"""Render docs/PHASE_1_2_REPORT.md → docs/PHASE_1_2_REPORT.pdf.

Same proven pipeline as `generate_explanation_pdf.py`. Only behavioural change:
**every `## ` heading starts on a new page** (the user's explicit requirement).
"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "PHASE_1_2_REPORT.md"
DST = ROOT / "docs" / "PHASE_1_2_REPORT.pdf"


# ----------------------------------------------------------------------------- styles

styles = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "Title", parent=styles["Title"], fontSize=26, leading=32,
    textColor=colors.HexColor("#264653"), alignment=1, spaceAfter=14,
)
SUBTITLE = ParagraphStyle(
    "SubTitle", parent=styles["Normal"], fontSize=16, leading=20,
    textColor=colors.HexColor("#2a9d8f"), alignment=1, spaceAfter=8,
)
META = ParagraphStyle(
    "Meta", parent=styles["Normal"], fontSize=11, leading=14,
    textColor=colors.HexColor("#264653"), alignment=1, spaceAfter=4,
)
H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=22, leading=26,
    textColor=colors.HexColor("#264653"), spaceBefore=0, spaceAfter=14,
    keepWithNext=True,
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=18, leading=22,
    textColor=colors.HexColor("#2a9d8f"), spaceBefore=0, spaceAfter=12,
    keepWithNext=True,
)
H3 = ParagraphStyle(
    "H3", parent=styles["Heading3"], fontSize=13, leading=17,
    textColor=colors.HexColor("#287271"), spaceBefore=12, spaceAfter=8,
    keepWithNext=True,
)
BODY = ParagraphStyle(
    "Body", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=8,
    textColor=colors.HexColor("#1d1d1d"),
)
QUOTE = ParagraphStyle(
    "Quote", parent=BODY, leftIndent=24, rightIndent=24, fontName="Helvetica-Oblique",
    textColor=colors.HexColor("#444444"), spaceBefore=4, spaceAfter=10,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY, leftIndent=18, bulletIndent=4, spaceAfter=4,
)
TABLE_HEADER = ParagraphStyle(
    "TableHeader", parent=BODY, fontSize=10, leading=12, textColor=colors.white,
    fontName="Helvetica-Bold", alignment=1, spaceAfter=0,
)
TABLE_CELL = ParagraphStyle(
    "TableCell", parent=BODY, fontSize=9.5, leading=12, spaceAfter=0,
)


# ----------------------------------------------------------------------------- inline transforms

def _inline(text: str) -> str:
    text = text.replace("&", "&amp;")
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(
        r"`([^`]+)`",
        r'<font face="Courier" backColor="#f0f0f0">\1</font>',
        text,
    )
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<link href="\2" color="#2a9d8f"><u>\1</u></link>',
        text,
    )
    return text


# ----------------------------------------------------------------------------- markdown → flowables

def parse_markdown(md: str) -> list:
    flow: list = []
    lines = md.splitlines()
    i = 0

    # Cover page = everything before the first '---' on its own line
    cover: list[str] = []
    while i < len(lines) and lines[i].strip() != "---":
        cover.append(lines[i])
        i += 1
    if i < len(lines):
        i += 1

    cover_text = "\n".join(cover).strip()
    cover_lines = [l for l in cover_text.split("\n") if l.strip()]
    if cover_lines:
        flow.append(Spacer(1, 1.5 * inch))
        title_line = next((l for l in cover_lines if l.startswith("# ")), "")
        if title_line:
            flow.append(Paragraph(_inline(title_line[2:]), TITLE))
            flow.append(Spacer(1, 0.15 * inch))
        # Subtitle = first '## ' under the title
        sub_line = next((l for l in cover_lines if l.startswith("## ")), "")
        if sub_line:
            flow.append(Paragraph(_inline(sub_line[3:]), SUBTITLE))
            flow.append(Spacer(1, 0.4 * inch))
        # Then the meta lines (anything else)
        for l in cover_lines:
            if l.startswith("# ") or l.startswith("## "):
                continue
            flow.append(Paragraph(_inline(l), META))
        flow.append(Spacer(1, 1.0 * inch))
        flow.append(Paragraph(
            '<font color="#888888"><i>Research / educational artefact. Not for clinical use.</i></font>',
            ParagraphStyle("Disclaimer", parent=BODY, alignment=1, fontSize=9),
        ))
        flow.append(PageBreak())

    # Main body — every H2 starts on a new page
    while i < len(lines):
        line = lines[i]

        # Horizontal rule → spacing only (page break is on the next H2)
        if line.strip() == "---":
            flow.append(Spacer(1, 0.05 * inch))
            i += 1
            continue

        # Headings — H2 always on a new page; H1 same; H3 inline
        if line.startswith("### "):
            flow.append(Paragraph(_inline(line[4:]), H3))
            i += 1
            continue
        if line.startswith("## "):
            flow.append(PageBreak())
            flow.append(Paragraph(_inline(line[3:]), H2))
            i += 1
            continue
        if line.startswith("# "):
            flow.append(PageBreak())
            flow.append(Paragraph(_inline(line[2:]), H1))
            i += 1
            continue

        # Block quote
        if line.startswith("> "):
            block: list[str] = []
            while i < len(lines) and lines[i].startswith("> "):
                block.append(lines[i][2:].rstrip())
                i += 1
            flow.append(Paragraph(_inline(" ".join(block)), QUOTE))
            continue

        # Table
        if line.lstrip().startswith("|") and i + 1 < len(lines) \
                and re.match(r"^\s*\|[\s|:\-]+\|\s*$", lines[i + 1]):
            tbl_lines: list[str] = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                tbl_lines.append(lines[i].strip())
                i += 1
            flow.append(_render_table(tbl_lines))
            flow.append(Spacer(1, 6))
            continue

        # Bullet list (with one level of nesting via 3+ leading spaces)
        if line.startswith("- ") or line.startswith("* "):
            while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("* ")
                                       or re.match(r"^   +(- |\* )", lines[i])):
                if re.match(r"^   +(- |\* )", lines[i]):
                    content = re.sub(r"^\s+(- |\* )", "", lines[i])
                    sub_style = ParagraphStyle("SubBullet", parent=BULLET, leftIndent=42, bulletIndent=28)
                    flow.append(Paragraph("◦ " + _inline(content), sub_style))
                else:
                    flow.append(Paragraph("• " + _inline(lines[i][2:].rstrip()), BULLET))
                i += 1
            continue

        # Numbered list
        if re.match(r"^\d+\.\s", line):
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i]):
                m = re.match(r"^(\d+)\.\s+(.*)$", lines[i])
                if m:
                    flow.append(Paragraph(f"{m.group(1)}. {_inline(m.group(2))}", BULLET))
                i += 1
            continue

        # Paragraph
        if line.strip():
            para_lines: list[str] = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", ">", "|", "- ", "* ")) \
                    and not re.match(r"^\d+\.\s", lines[i]) and lines[i].strip() != "---":
                para_lines.append(lines[i].rstrip())
                i += 1
            flow.append(Paragraph(_inline(" ".join(para_lines)), BODY))
            continue

        i += 1

    return flow


def _render_table(tbl_lines: list[str]) -> Table:
    rows: list[list[str]] = []
    for ln in tbl_lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        rows.append(cells)
    if len(rows) >= 2 and re.match(r"^[:\-\s]+$", "".join(rows[1])):
        rows.pop(1)
    body: list[list[Paragraph]] = []
    for ri, row in enumerate(rows):
        style = TABLE_HEADER if ri == 0 else TABLE_CELL
        body.append([Paragraph(_inline(c), style) for c in row])

    n_cols = max(len(r) for r in body)
    avail_width = LETTER[0] - 1.7 * inch
    col_widths = [avail_width / n_cols] * n_cols

    tbl = Table(body, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#264653")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f7faf9"), colors.white]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#264653")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    return tbl


def main() -> None:
    md_text = SRC.read_text(encoding="utf-8")
    flow = parse_markdown(md_text)

    doc = SimpleDocTemplate(
        str(DST),
        pagesize=LETTER,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.95 * inch,
        bottomMargin=0.95 * inch,
        title="NeuroVal-3D - Complete Phase 1 + Phase 2 Report",
        author="Naveen Rajdev, Pooja P, Vikneshwaran Marimuthu, Vaishnavi Pagad",
    )

    def add_page_furniture(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(
            0.85 * inch, 0.55 * inch,
            "NeuroVal-3D - B.Tech Minor Project - April 2026",
        )
        canvas.drawRightString(
            LETTER[0] - 0.85 * inch, 0.55 * inch,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    doc.build(flow, onFirstPage=add_page_furniture, onLaterPages=add_page_furniture)
    print(f"wrote {DST}")
    print(f"  size: {DST.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
