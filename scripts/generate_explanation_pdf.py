"""Render docs/PROJECT_EXPLAINED.md to docs/PROJECT_EXPLAINED.pdf.

Custom-styled output (cover page, section headings, tables, body text). Uses reportlab's
Platypus high-level layout. The markdown source remains the single source of truth — this
script only converts.
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
SRC = ROOT / "docs" / "PROJECT_EXPLAINED.md"
DST = ROOT / "docs" / "PROJECT_EXPLAINED.pdf"

# ----------------------------------------------------------------------------- styles

styles = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "Title", parent=styles["Title"], fontSize=24, leading=30,
    textColor=colors.HexColor("#264653"), alignment=1, spaceAfter=12,
)
SUBTITLE = ParagraphStyle(
    "SubTitle", parent=styles["Normal"], fontSize=14, leading=18,
    textColor=colors.HexColor("#2a9d8f"), alignment=1, spaceAfter=6,
)
META = ParagraphStyle(
    "Meta", parent=styles["Normal"], fontSize=11, leading=14,
    textColor=colors.HexColor("#264653"), alignment=1, spaceAfter=4,
)
H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=18, leading=22,
    textColor=colors.HexColor("#264653"), spaceBefore=18, spaceAfter=10,
    keepWithNext=True,
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=14, leading=18,
    textColor=colors.HexColor("#2a9d8f"), spaceBefore=12, spaceAfter=8,
    keepWithNext=True,
)
H3 = ParagraphStyle(
    "H3", parent=styles["Heading3"], fontSize=12, leading=16,
    textColor=colors.HexColor("#287271"), spaceBefore=8, spaceAfter=6,
    keepWithNext=True,
)
BODY = ParagraphStyle(
    "Body", parent=styles["Normal"], fontSize=10.5, leading=14, spaceAfter=8,
    textColor=colors.HexColor("#1d1d1d"),
)
QUOTE = ParagraphStyle(
    "Quote", parent=BODY, leftIndent=20, rightIndent=20, fontName="Helvetica-Oblique",
    textColor=colors.HexColor("#444444"), spaceBefore=4, spaceAfter=8,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY, leftIndent=18, bulletIndent=4, spaceAfter=4,
)
TABLE_HEADER = ParagraphStyle(
    "TableHeader", parent=BODY, fontSize=10, leading=12, textColor=colors.white,
    fontName="Helvetica-Bold", alignment=1,
)
TABLE_CELL = ParagraphStyle(
    "TableCell", parent=BODY, fontSize=9.5, leading=12, spaceAfter=0,
)


# ----------------------------------------------------------------------------- inline transforms

def _inline(text: str) -> str:
    """Convert markdown inline syntax → reportlab mini-html."""
    # Escape ampersand first
    text = text.replace("&", "&amp;")
    # Bold **x** then italic *x*
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", text)
    # Inline code `x`
    text = re.sub(
        r"`([^`]+)`",
        r'<font face="Courier" backColor="#f0f0f0">\1</font>',
        text,
    )
    # Links [text](url)
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

    # ----- detect cover-page block (everything before the first '---' on its own line at top)
    cover: list[str] = []
    while i < len(lines) and lines[i].strip() != "---":
        cover.append(lines[i])
        i += 1
    if i < len(lines):
        i += 1  # skip the '---'

    # Render the cover
    cover_text = "\n".join(cover).strip()
    cover_lines = [l for l in cover_text.split("\n") if l.strip()]
    if cover_lines:
        # Title is first '# ' line
        title_line = next((l for l in cover_lines if l.startswith("# ")), "")
        if title_line:
            flow.append(Spacer(1, 1.5 * inch))
            flow.append(Paragraph(_inline(title_line[2:]), TITLE))
            flow.append(Spacer(1, 0.2 * inch))
        for l in cover_lines[1:]:
            if l.startswith("**") and l.endswith("**"):
                flow.append(Paragraph(_inline(l), SUBTITLE))
            else:
                flow.append(Paragraph(_inline(l), META))
        flow.append(Spacer(1, 1.0 * inch))
        flow.append(Paragraph(
            '<font color="#999999"><i>This is a research/educational artefact. Not for clinical use.</i></font>',
            ParagraphStyle("Disclaimer", parent=BODY, alignment=1, fontSize=9),
        ))
        flow.append(PageBreak())

    # ----- main body
    while i < len(lines):
        line = lines[i]

        # Horizontal rule → page break
        if line.strip() == "---":
            flow.append(Spacer(1, 0.05 * inch))
            i += 1
            continue

        # Headings
        if line.startswith("### "):
            flow.append(Paragraph(_inline(line[4:]), H3)); i += 1; continue
        if line.startswith("## "):
            flow.append(Paragraph(_inline(line[3:]), H2)); i += 1; continue
        if line.startswith("# "):
            flow.append(Paragraph(_inline(line[2:]), H1)); i += 1; continue

        # Block quote (>)
        if line.startswith("> "):
            block: list[str] = []
            while i < len(lines) and lines[i].startswith("> "):
                block.append(lines[i][2:].rstrip())
                i += 1
            flow.append(Paragraph(_inline(" ".join(block)), QUOTE))
            continue

        # Table — heuristic: line starting with | and the next line is a separator with --- in cells
        if line.lstrip().startswith("|") and i + 1 < len(lines) \
                and re.match(r"^\s*\|[\s|:\-]+\|\s*$", lines[i + 1]):
            tbl_lines: list[str] = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                tbl_lines.append(lines[i].strip())
                i += 1
            flow.append(_render_table(tbl_lines))
            flow.append(Spacer(1, 6))
            continue

        # Bullet list
        if line.startswith("- ") or line.startswith("* "):
            while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("* ")):
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

        # Paragraph (collect contiguous non-empty lines)
        if line.strip():
            para_lines: list[str] = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", ">", "|", "- ", "* ")) \
                    and not re.match(r"^\d+\.\s", lines[i]) and lines[i].strip() != "---":
                para_lines.append(lines[i].rstrip())
                i += 1
            flow.append(Paragraph(_inline(" ".join(para_lines)), BODY))
            continue

        # Blank line
        i += 1

    return flow


def _render_table(tbl_lines: list[str]) -> Table:
    """Convert markdown table lines → reportlab Table."""
    rows: list[list[str]] = []
    for ln in tbl_lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        rows.append(cells)
    if len(rows) >= 2 and re.match(r"^[:\-\s]+$", "".join(rows[1])):
        rows.pop(1)  # drop separator
    # Wrap cells in Paragraphs so they handle wrapping
    body: list[list[Paragraph]] = []
    for ri, row in enumerate(rows):
        style = TABLE_HEADER if ri == 0 else TABLE_CELL
        body.append([Paragraph(_inline(c), style) for c in row])

    n_cols = max(len(r) for r in body)
    avail_width = LETTER[0] - 1.5 * inch
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
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


# ----------------------------------------------------------------------------- main

def main() -> None:
    md_text = SRC.read_text(encoding="utf-8")
    flow = parse_markdown(md_text)

    doc = SimpleDocTemplate(
        str(DST),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title="NeuroVal-3D — A Lie Detector for AI-Written Brain MRI Reports",
        author="Naveen Rajdev, Pooja P, Vikneshwaran Marimuthu, Vaishnavi Pagad",
    )

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(
            0.75 * inch, 0.5 * inch,
            "NeuroVal-3D · B.Tech Minor Project · April 2026",
        )
        canvas.drawRightString(
            LETTER[0] - 0.75 * inch, 0.5 * inch,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    doc.build(flow, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"wrote {DST}")
    print(f"  size: {DST.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
