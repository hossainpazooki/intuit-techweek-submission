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

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "submission" / "submission_D_writeup.md"
OUT = ROOT / "submission" / "submission_D_writeup.pdf"

# Helvetica (a built-in PDF font) lacks many math/typographic glyphs; map them to
# readable ASCII so the PDF renders cleanly without registering a Unicode TTF.
UNICODE_MAP = {
    "≤": "<=", "≥": ">=", "≈": "~=", "≠": "!=",
    # NB: the multiplication dot maps to a SPACED asterisk on purpose -- a bare
    # "*" collides with the italic markdown regex below (it would pair with a
    # genuine *emphasis* marker across a code span and emit invalid XML); a
    # space-padded "*" can neither open nor close italics, so it stays literal.
    "→": "->", "×": "x", "·": " * ", "π": "pi",
    "∏": "prod", "∑": "sum", "√": "sqrt", "∈": "in",
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
    # References are excluded from the 4-page body limit and the 11pt-body rule.
    refs = ParagraphStyle("refs", parent=body, fontSize=9, leading=10.5, spaceBefore=2)

    story, bullets, para = [], [], []
    state = {"refs": False}

    def flush_para():
        if para:
            story.append(Paragraph(inline(" ".join(para)),
                                   refs if state["refs"] else body))
            para.clear()

    def flush_bullets():
        if bullets:
            items = [ListItem(Paragraph(inline(b), body), leftIndent=12)
                     for b in bullets]
            story.append(ListFlowable(items, bulletType="bullet", start="•",
                                      leftIndent=14, bulletFontSize=9, spaceBefore=1,
                                      spaceAfter=4))
            bullets.clear()

    for ln in lines:
        s = ln.strip()
        if not s:
            flush_para(); flush_bullets(); continue
        if s.startswith("# "):
            flush_para(); flush_bullets(); story.append(Paragraph(inline(s[2:]), h1))
        elif s.startswith("## "):
            flush_para(); flush_bullets(); story.append(Paragraph(inline(s[3:]), h2))
        elif s.startswith("- "):
            flush_para(); bullets.append(s[2:])
        elif s.startswith("**Team:**"):
            flush_para(); flush_bullets()
            story.append(Paragraph(inline(s), team))
        elif s.startswith("**References."):
            flush_para(); flush_bullets(); state["refs"] = True; para.append(s)
        else:
            flush_bullets(); para.append(s)
    flush_para(); flush_bullets()

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
