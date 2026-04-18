"""Build a side-by-side practice PDF.

Left half  : slide thumbnail
Right half : matching script text

Usage:
    python script/build_practice_pdf.py
"""

import re
import glob
from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

SCRIPT_TXT = Path(__file__).parent / "full_presentation_script.txt"
SLIDES_DIR = Path("/tmp")
OUTPUT_PDF = Path(__file__).parent / "practice_script_with_slides.pdf"

# Maps script slide number → actual PNG file index (sequential export order)
# PNG 001=Title, 002=Agenda, 003=Section:Intro cover, 004=Slide2(Background),
# 005=Slide3(Constraint), 006=Slide4(ColdStart), 007=Slide6(Hook),
# 008=Slide7(RQ), 009=Slide8(Contributions), 010=Section:RelatedWork,
# 011=Slide9(Embeddings), 012=Slide10(CNN/Transformer), 013=Section:Method,
# 014=Slide12(Pipeline), 015=Slide13(FormalSetup), 016=Slide15(Models),
# 017=Slide16(ProxyTasks), 018=Slide17(Labels), 019=Slide18(Protocol),
# 020=Section:Results, 021=Slide19(Composer), 022=Slide21(Character),
# 023=Slide22(MetricMismatch), 024=Slide24(ROI), 025=Section:Discussion,
# 026=Slide25(Discussion), 027=Slide26(SystemImplications),
# 028=Slide27(Limitations), 029=Slide28(FutureWork), 030=Section:Conclusion,
# 031=Slide29(Conclusion), 032=Slide30(References), 033=Slide31(ThankYou)
SCRIPT_TO_PNG = {
    1:  1,   # Title
    2:  2,   # Agenda
    3:  3,   # Section: Introduction cover
    4:  4,   # Slide 2: Where This Research Comes From
    5:  5,   # Slide 3: Real-World Constraint
    6:  6,   # Slide 4: Cold-Start Problem
    7:  7,   # Slide 6: Hook
    8:  8,   # Slide 7: Research Questions
    9:  9,   # Slide 8: Contributions
    10: 10,  # Section: Related Work cover
    11: 11,  # Slide 9: Why Embeddings
    12: 12,  # Slide 10: CNN vs Transformer
    13: 13,  # Section: Method cover
    14: 14,  # Slide 12: Pipeline Overview
    15: 15,  # Slide 13: Formal Setup
    16: 16,  # Slide 15: Models Evaluated
    17: 17,  # Slide 16: Proxy Tasks
    18: 18,  # Slide 17: Label Construction
    19: 19,  # Slide 18: Evaluation Protocol
    20: 20,  # Section: Results cover
    21: 21,  # Slide 19: Composer Retrieval
    22: 22,  # Slide 21: Character Retrieval
    23: 23,  # Slide 22: Metric Mismatch
    24: 24,  # Slide 24: ROI
    25: 25,  # Section: Discussion cover
    26: 26,  # Slide 25: Discussion
    27: 27,  # Slide 26: System Implications
    28: 28,  # Slide 27: Limitations
    29: 29,  # Slide 28: Future Work
    30: 30,  # Section: Conclusion cover
    31: 31,  # Slide 29: Conclusion
    32: 32,  # Slide 30: References
    33: 33,  # Slide 31: Thank You
}

# Script numbers used in the txt file (comment numbers, not sequential)
# Maps script comment number → sequential page index
COMMENT_TO_PNG = {
    # script_num : png_file_index
    5:  7,   # Hook
    6:  8,   # Research Questions
    7:  9,   # Three Contributions
    9:  11,  # Why Embeddings Matter
    10: 12,  # CNN vs Transformer
    12: 14,  # Pipeline Overview
    13: 15,  # Formal Problem Setup
    14: 16,  # Models Evaluated
    15: 17,  # Two Proxy Tasks
    16: 18,  # Label Construction
    17: 19,  # Evaluation Protocol
    19: 21,  # Composer Retrieval
    20: 22,  # Character Retrieval
    21: 23,  # Metric Mismatch
    22: 24,  # ROI
    24: 26,  # Discussion: What This Means
    26: 28,  # Limitations
    27: 29,  # Future Work
    29: 31,  # Conclusion
}

GREEN = HexColor("#1a4d2e")
GOLD  = HexColor("#c9a84c")
PAGE_W, PAGE_H = landscape(A4)   # 841.89 x 595.28 pt
MARGIN = 1.2 * cm
SLIDE_W = PAGE_W / 2 - MARGIN * 1.5
SLIDE_H = SLIDE_W * 9 / 16       # 16:9 aspect

# ── Parse script into per-slide chunks ────────────────────────────────────────

def parse_script(txt_path: Path) -> list[dict]:
    """Return list of {slide_num, header, body} dicts in slide order."""
    text = txt_path.read_text(encoding="utf-8")
    # Split on slide markers: --- Slide N: ... --- or --- Slide N (no colon)
    pattern = re.compile(r"--- (Slide \d+[^-]*?) ---")
    parts = pattern.split(text)
    # parts: [preamble, header1, body1, header2, body2, ...]

    # Also capture section cover markers
    section_pattern = re.compile(
        r"--- (Slide \d+: \[Section Cover[^\]]*\][^-]*) ---"
    )

    chunks = []
    # Gather section/preamble text before first slide marker
    preamble = parts[0].strip()

    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body   = parts[i + 1].strip() if i + 1 < len(parts) else ""
        # Extract slide number
        m = re.match(r"Slide (\d+)", header)
        num = int(m.group(1)) if m else 0
        chunks.append({"slide_num": num, "header": header, "body": body})

    return chunks


def find_slide_image(png_index: int) -> Path | None:
    # Marp exports without extension: slides.001, slides.002, ...
    for pattern in [
        str(SLIDES_DIR / f"slides.{png_index:03d}"),
        str(SLIDES_DIR / f"slides.{png_index:03d}.png"),
    ]:
        matches = glob.glob(pattern)
        if matches:
            return Path(matches[0])
    return None


# ── Draw one page (slide top, script bottom) using canvas directly ───────────

def draw_page(c: canvas.Canvas, slide_img_path: Path | None,
              header: str, body: str, page_num: int) -> None:
    c.setPageSize(A4)           # portrait for vertical stacking
    w, h = A4

    pad = MARGIN
    top_bar = 10

    # ── top bar ──
    c.setFillColor(GREEN)
    c.rect(0, h - top_bar, w, top_bar, fill=1, stroke=0)
    c.setFillColor(GOLD)
    c.rect(w * 0.7, h - top_bar, w * 0.3, top_bar, fill=1, stroke=0)

    # ── TOP HALF: slide image ──
    slide_area_h = h * 0.40          # top ~40% for slide
    slide_y_top  = h - top_bar - pad  # top of slide area
    slide_y_bot  = slide_y_top - slide_area_h

    if slide_img_path and slide_img_path.exists():
        img = Image.open(slide_img_path)
        iw, ih = img.size
        ratio = ih / iw                      # 9/16 ≈ 0.5625
        draw_w = w - pad * 2
        draw_h = draw_w * ratio
        # centre vertically in slide area
        y_img = slide_y_bot + (slide_area_h - draw_h) / 2
        c.drawImage(str(slide_img_path), pad, y_img, draw_w, draw_h,
                    preserveAspectRatio=True)
    else:
        c.setFillColor(HexColor("#eeeeee"))
        c.rect(pad, slide_y_bot, w - pad * 2, slide_area_h, fill=1, stroke=0)
        c.setFillColor(HexColor("#aaaaaa"))
        c.setFont("Helvetica", 12)
        c.drawCentredString(w / 2, slide_y_bot + slide_area_h / 2, "[No slide image]")

    # ── divider ──
    c.setStrokeColor(GOLD)
    c.setLineWidth(1)
    c.line(pad, slide_y_bot - 6, w - pad, slide_y_bot - 6)

    # ── BOTTOM HALF: blank space for handwritten notes ──
    # page number only
    c.setFillColor(HexColor("#888888"))
    c.setFont("Helvetica", 8)
    c.drawRightString(w - pad, pad / 2, str(page_num))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    chunks = parse_script(SCRIPT_TXT)
    print(f"[INFO] Parsed {len(chunks)} slide script chunks")

    # Build script lookup by comment slide number
    script_map = {ch["slide_num"]: ch for ch in chunks}

    # Pages with no script body — labels only
    section_covers = {
        3:  "Section Cover: Introduction\n(improvise — narrative bridge to research)",
        6:  "Slide 4: The Cold-Start Problem in Music RecSys\n\nSo, when we talk about building a recommendation system for iPalpiti,\nthe very first obstacle we run into is what's called the cold-start problem.\n\nThere is no user history. No listening logs, no ratings, no 'people who liked\nthis also liked that.' Collaborative filtering, which powers most modern\nrecommendation systems, simply has nothing to work with.\n\nWhat we do have is the audio itself — over 200 tracks, each one a rich, full\nrecording. Candidate generation must rely entirely on content.\n\nThe implication is stark: embedding quality equals ranking quality. No fallback.",
        10: "Section Cover: Related Work\n(pause briefly)",
        13: "Section Cover: Method\n(pause briefly)",
        20: "Section Cover: Results\n(pause briefly)",
        25: "Section Cover: Discussion\n(pause briefly)",
        27: "Slide 25: System-Level Implications\n\nFor anyone designing a real cold-start recommendation system,\nthese findings have direct operational meaning.\n\nIn cold-start settings, the embedding model IS the ranking system.\nNo collaborative signal to compensate for poor candidates.\nModel selection is a direct operational decision.\n\nMid-sized models offer the best cost-quality profile.\nCNN-Small / CNN-Medium: within 0.04 NDCG of Transformer-Medium\nat roughly 10x lower extraction latency.\n\nFor iPalpiti — no GPU infrastructure, frequent catalog updates —\na 25x extraction overhead needs a real ranking gain to justify it.\nWe found no such justification.",
        30: "Section Cover: Conclusion\n(pause briefly)",
    }

    c = canvas.Canvas(str(OUTPUT_PDF), pagesize=A4)

    # Iterate all 33 PNG pages sequentially
    for png_idx in range(1, 34):
        img_path = find_slide_image(png_idx)

        # Find matching script chunk using COMMENT_TO_PNG reverse lookup
        script_comment_num = None
        for comment_num, mapped_png in COMMENT_TO_PNG.items():
            if mapped_png == png_idx:
                script_comment_num = comment_num
                break

        if script_comment_num and script_comment_num in script_map:
            ch = script_map[script_comment_num]
            header = ch["header"]
            body   = ch["body"]
        elif png_idx in section_covers:
            header = f"PNG {png_idx}"
            body   = section_covers[png_idx]
        elif png_idx == 1:
            header = "Slide 1: Title"
            body   = "(improvise — introduce yourself and the title)"
        elif png_idx == 2:
            header = "Agenda"
            body   = "(walk through the agenda briefly — no script needed)"
        elif png_idx == 32:
            header = "References"
            body   = "(no script — available for Q&A reference)"
        elif png_idx == 33:
            header = "Thank You"
            body   = "Thank you. I am happy to take any questions."
        else:
            header = f"PNG {png_idx}"
            body   = "(no script)"

        draw_page(c, img_path, header, body, png_idx)
        c.showPage()

    c.save()
    print(f"[SAVED] {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
