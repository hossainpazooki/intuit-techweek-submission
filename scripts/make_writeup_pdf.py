"""Render submission/submission_D_writeup.md -> submission/submission_D_writeup.pdf.

Self-contained markdown->PDF for the Deliverable D writeup, honoring the enforced
format: US Letter, >= 0.75in margins, 11pt body font, the five required headers in
order. sklearn-stack-friendly: depends only on reportlab (no system libraries, no
LaTeX/pandoc). The markdown stays the source of record; this is the export step.

Usage:
    python scripts/make_writeup_pdf.py
"""

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "submission" / "submission_D_writeup.md"
OUT = ROOT / "submission" / "submission_D_writeup.pdf"

# Usable text width on US Letter with 0.75in margins (8.5 - 1.5 = 7.0in).
CONTENT_WIDTH = 7.0 * inch

# Helvetica (a built-in PDF font) lacks many math/typographic glyphs; map them to
# readable ASCII so the PDF renders cleanly without registering a Unicode TTF.
UNICODE_MAP = {
    "≤": "<=", "≥": ">=", "≈": "~=", "≠": "!=",
    # NB: the multiplication dot maps to a SPACED asterisk on purpose -- a bare
    # "*" collides with the italic markdown regex below (it would pair with a
    # genuine *emphasis* marker across a code span and emit invalid XML); a
    # space-padded "*" can neither open nor close italics, so it stays literal.
    "→": "->", "×": "x", "·": " * ", "π": "pi",
    "∏": "prod", "∑": "sum", "Σ": "sum", "√": "sqrt", "∈": "in", "±": "+/-",
    "—": " - ", "–": "-", "…": "...", "′": "'",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    " ": " ", "≡": "==", "≈": "~=",
}


def deunicode(s: str) -> str:
    for k, v in UNICODE_MAP.items():
        s = s.replace(k, v)
    # remaining glyphs Helvetica lacks, addressed by codepoint to stay ASCII-safe:
    for cp, rep in ((0x2212, "-"), (0x00A7, "Section "), (0x0302, ""), (0x00A0, " ")):
        s = s.replace(chr(cp), rep)
    # subscripts used in the survival math (S_{t}, h_d, X_{-f}) -> plain ASCII
    s = s.replace("₀", "0").replace("₁", "1")
    return s.encode("ascii", "replace").decode("ascii")


def inline(s: str) -> str:
    """Convert a small markdown subset to reportlab's mini-HTML and escape XML."""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    return s


def main() -> int:
    raw = SRC.read_text(encoding="utf-8")
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)  # drop HTML comments
    lines = [deunicode(ln.rstrip()) for ln in raw.splitlines()]

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=11, leading=12.4, alignment=TA_LEFT,
                          spaceAfter=3, spaceBefore=0)
    h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=13.5,
                        leading=15.5, spaceAfter=3, spaceBefore=0, keepWithNext=True)
    h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=12,
                        leading=13.5, spaceBefore=6, spaceAfter=2, keepWithNext=True)
    team = ParagraphStyle("team", parent=body, fontSize=11, spaceAfter=4)
    quote = ParagraphStyle("quote", parent=body, fontName="Helvetica-Oblique",
                           leftIndent=12, spaceAfter=4)
    # References are excluded from the 4-page body limit and the 11pt-body rule.
    refs = ParagraphStyle("refs", parent=body, fontSize=9, leading=10.5, spaceBefore=2)
    caption = ParagraphStyle("caption", parent=body, fontName="Helvetica-Oblique",
                             fontSize=8.5, leading=10, alignment=TA_CENTER,
                             spaceBefore=2, spaceAfter=6)
    # Tables/figures are not body prose; a smaller cell font keeps them on-page
    # without touching the 11pt minimum that governs the body text.
    cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=10,
                          spaceAfter=0, spaceBefore=0)
    cellh = ParagraphStyle("cellh", parent=cell, fontName="Helvetica-Bold")

    story, bullets, para, quotes, tbl = [], [], [], [], []
    state = {"refs": False}

    def emit_image(alt: str, rel: str):
        """Render a Markdown image, scaled to fit the text column, with a caption."""
        path = (ROOT / rel).resolve()
        if not path.exists():
            story.append(Paragraph(inline(f"[missing figure: {rel}]"), body))
            return
        iw, ih = ImageReader(str(path)).getSize()
        w = min(CONTENT_WIDTH * 0.56, iw)  # compact figures: substance over size
        h = w * ih / iw
        img = Image(str(path), width=w, height=h)
        img.hAlign = "CENTER"
        flow = [img]
        if alt.strip():
            flow.append(Paragraph(inline(alt.strip()), caption))
        story.append(KeepTogether(flow))

    def flush_table():
        if not tbl:
            return
        # tbl is a list of raw "| a | b |" lines; the 2nd is the |---|---| rule.
        def cells(line):
            parts = line.strip().strip("|").split("|")
            return [p.strip() for p in parts]
        rows = [cells(t) for t in tbl]
        rows = [r for i, r in enumerate(rows)
                if not (i == 1 and all(set(c) <= set(":- ") for c in r))]
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        # Column widths proportional to the longest cell text, so the wide
        # "what changed" columns get the room and narrow numeric ones stay tight.
        spans = [max(len(rows[i][c]) for i in range(len(rows))) for c in range(ncol)]
        total = sum(spans) or 1
        widths = [max(0.55 * inch, CONTENT_WIDTH * s / total) for s in spans]
        scale = CONTENT_WIDTH / sum(widths)
        widths = [w * scale for w in widths]
        data = [[Paragraph(inline(c), cellh if r == 0 else cell)
                 for c in row] for r, row in enumerate(rows)]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b0b0")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(Spacer(1, 2))
        story.append(t)
        story.append(Spacer(1, 4))
        tbl.clear()

    def flush_para():
        if para:
            story.append(Paragraph(inline(" ".join(para)),
                                   refs if state["refs"] else body))
            para.clear()

    def flush_quote():
        if quotes:
            story.append(Paragraph(inline(" ".join(quotes)), quote))
            quotes.clear()

    def flush_bullets():
        if bullets:
            items = [ListItem(Paragraph(inline(b), body), leftIndent=12)
                     for b in bullets]
            story.append(ListFlowable(items, bulletType="bullet", start="•",
                                      leftIndent=14, bulletFontSize=9, spaceBefore=1,
                                      spaceAfter=4))
            bullets.clear()

    def flush_all():
        flush_para(); flush_bullets(); flush_quote(); flush_table()

    img_re = re.compile(r"^!\[(.*?)\]\((.+?)\)$")

    for ln in lines:
        s = ln.strip()
        if not s:
            flush_all(); continue
        img = img_re.match(s)
        if s.startswith("|") and s.endswith("|"):
            flush_para(); flush_bullets(); flush_quote(); tbl.append(s)
        elif img:
            flush_all(); emit_image(img.group(1), img.group(2))
        elif s.startswith("# "):
            flush_all()
            story.append(Paragraph(inline(s[2:]), h1))
        elif s.startswith("## "):
            flush_all()
            story.append(Paragraph(inline(s[3:]), h2))
        elif s == ">" or s.startswith("> "):
            flush_para(); flush_bullets(); flush_table()
            q = s[1:].lstrip()
            if q:
                quotes.append(q)
            else:  # bare ">" separates blockquote paragraphs
                flush_quote()
        elif s.startswith("- "):
            flush_para(); flush_quote(); flush_table(); bullets.append(s[2:])
        elif s.startswith("**Team:**"):
            flush_all()
            story.append(Paragraph(inline(s), team))
        elif s.startswith("**References."):
            flush_all()
            state["refs"] = True; para.append(s)
        else:
            flush_bullets(); flush_quote(); flush_table(); para.append(s)
    flush_all()

    doc = SimpleDocTemplate(str(OUT), pagesize=letter,
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                            title="Deliverable D - Technical Writeup")
    doc.build(story)
    print(f"wrote {OUT.relative_to(ROOT)}  ({doc.page} page(s), body 11pt, 0.75in margins)")
    if doc.page > 4:
        print("WARNING: exceeds the enforced 4-page body limit -- trim before upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
