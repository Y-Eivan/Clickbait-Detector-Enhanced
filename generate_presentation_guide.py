from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

OUTPUT = "Presentation_and_Poster_Guide.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)

# ── Colours ──────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1E2761")
TEAL   = colors.HexColor("#028090")
LIGHT  = colors.HexColor("#E8F4F8")
GRAY   = colors.HexColor("#555555")
WHITE  = colors.white
BLACK  = colors.black

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

doc_title = ParagraphStyle(
    "DocTitle", parent=styles["Title"],
    fontSize=22, textColor=NAVY, spaceAfter=6, alignment=TA_CENTER, leading=28
)
doc_subtitle = ParagraphStyle(
    "DocSubtitle", parent=styles["Normal"],
    fontSize=11, textColor=GRAY, spaceAfter=20, alignment=TA_CENTER
)
section_header = ParagraphStyle(
    "SectionHeader", parent=styles["Heading1"],
    fontSize=15, textColor=WHITE, spaceAfter=0, spaceBefore=18,
    backColor=NAVY, leftIndent=-10, rightIndent=-10,
    borderPad=6, leading=20
)
slide_title = ParagraphStyle(
    "SlideTitle", parent=styles["Heading2"],
    fontSize=12, textColor=NAVY, spaceAfter=3, spaceBefore=12, leading=16
)
body = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=10, textColor=BLACK, spaceAfter=2, leading=14
)
bullet_style = ParagraphStyle(
    "Bullet", parent=styles["Normal"],
    fontSize=10, textColor=BLACK, spaceAfter=2, leftIndent=16, leading=14,
    bulletIndent=6
)
note_style = ParagraphStyle(
    "Note", parent=styles["Normal"],
    fontSize=9, textColor=GRAY, spaceAfter=4, leftIndent=16, leading=13, italic=True
)
poster_section = ParagraphStyle(
    "PosterSection", parent=styles["Heading2"],
    fontSize=12, textColor=WHITE, spaceAfter=0, spaceBefore=14,
    backColor=TEAL, leftIndent=-10, rightIndent=-10,
    borderPad=5, leading=18
)

def bullets(items):
    return [Paragraph(f"• {i}", bullet_style) for i in items]

def slide_block(number, title, content_items, note=None):
    elems = []
    elems.append(Paragraph(f"Slide {number}: {title}", slide_title))
    elems.extend(bullets(content_items))
    if note:
        elems.append(Paragraph(f"📌 {note}", note_style))
    return elems

story = []

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Clickbait Detector — Final Project", doc_title))
story.append(Paragraph("Presentation &amp; Poster Content Guide · RMCS Binus University", doc_subtitle))
story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=14))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: PRESENTATION SLIDES
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("  PART 1 — PRESENTATION SLIDES  ", section_header))
story.append(Spacer(1, 0.3*cm))

# Slide 1
story.extend(slide_block(1, "Title Slide", [
    "Title: Enhancing YouTube Clickbait Detection Using Comment Sentiment Analysis and Title-Transcript Semantic Similarity",
    "All team member names + student IDs",
    "Course: RMCS — Binus University, Semester 4",
    "Presentation date",
]))

# Slide 2
story.extend(slide_block(2, "Background / Motivation", [
    "YouTube has billions of videos — clickbait is a growing, widespread problem",
    "Users waste time on content that doesn't deliver what the title promises",
    "Existing detectors rely only on surface-level metadata (titles, view counts, likes)",
    "Two important signals are being ignored:",
    "  — What viewers say in comments (negative sentiment = disappointment)",
    "  — Whether the title actually matches the video content",
]))

# Slide 3
story.extend(slide_block(3, "Problem Statement", [
    "How can we improve YouTube clickbait detection by incorporating:",
    "  (1) Viewer comment sentiment, and",
    "  (2) Semantic mismatch between title and transcript?",
    "Research gap: No prior work combines VADER comment sentiment + Sentence-BERT title-to-transcript similarity in a single pipeline",
]))

# Slide 4
story.extend(slide_block(4, "Related Work", [
    "Vierti (2019) — SVM baseline model + open dataset (what this project builds on)",
    "Gothankar et al. (2022) — suggested title-transcript similarity as a clickbait signal",
    "Elyashar et al. (2022) — showed comment sentiment correlates with clickbait",
    "Bronakowski et al. (2023) — semantic features achieved 98% accuracy on news headlines",
    "Ahmadi & Chowanda (2023) — validated title-content matching for news clickbait",
], note="Keep to ~4-5 references. One sentence per reference is enough."))

# Slide 5
story.extend(slide_block(5, "Proposed System Overview (Pipeline Diagram)", [
    "Show a flowchart: Raw Dataset → Stage 1 (Extraction & Filtering) → Stage 2 (Feature Engineering: Block A, B, C) → Stage 3 (Training & Evaluation)",
    "Highlight the 3 feature blocks as the novel contribution",
], note="This should be a visual diagram, not a text slide. Use the flowchart from your report if available."))

# Slide 6
story.extend(slide_block(6, "Dataset", [
    "Source: Vierti's open YouTube dataset (~36,000 labeled videos)",
    "After filtering for available transcripts + comments: 9,393 videos",
    "After balancing 1:1 (clickbait : non-clickbait): 5,722 videos",
    "Train / Test split: 4,577 train (80%) / 1,145 test (20%) — stratified",
    "Data collection challenges:",
    "  — YouTube API daily quota limits → required API key rotation",
    "  — IP bans on transcript extraction → used cookie spoofing + exponential backoff",
]))

# Slide 7
story.extend(slide_block(7, "Feature Engineering — Block A (Baseline)", [
    "Word2Vec title embeddings (trained on dataset titles) → 25 dimensions",
    "  Params: vector_size=25, window=20, min_count=1, epochs=30",
    "Engagement metrics → 4 dimensions (log-scaled + min-max normalized):",
    "  log(1+views), log(1+likes), log(1+dislikes), log(1+comments)",
    "Total Block A: 29 features",
    "This replicates Vierti's original feature set (Model 1 baseline)",
]))

# Slide 8
story.extend(slide_block(8, "Feature Engineering — Block B (Comment Sentiment)", [
    "VADER Sentiment Analyzer applied to top 100 comments per video",
    "4 features extracted:",
    "  — mean_compound: average sentiment polarity",
    "  — std_compound: standard deviation of sentiment",
    "  — pct_positive: % comments with compound score > 0.3",
    "  — pct_negative: % comments with compound score < -0.3",
    "Rationale: Angry or disappointed viewers leave negative comments under clickbait videos",
]))

# Slide 9
story.extend(slide_block(9, "Feature Engineering — Block C (Title-Transcript Similarity)", [
    "Sentence-BERT (all-MiniLM-L6-v2) encodes both:",
    "  — Video title → embedding vector",
    "  — Video transcript → embedding vector",
    "Cosine similarity between title and transcript → 1 feature",
    "High similarity = title matches content (likely genuine)",
    "Low similarity = misleading title (likely clickbait)",
    "Note: Transcripts truncated at 256 subword tokens by the tokenizer",
]))

# Slide 10
story.extend(slide_block(10, "Models Compared", [
    "Model 1 — Baseline SVM: Block A only (29-dim), Vierti's original hyperparams (C=3.7, γ=4.1)",
    "Model 2 — Enhanced SVM: Blocks A+B+C (34-dim), tuned via grid search (C=10, γ=scale)",
    "Model 3 — Enhanced Random Forest: Blocks A+B+C (34-dim), tuned via grid search",
    "Hyperparameter tuning: 5-fold cross-validation optimizing F1-score",
    "Feature dimensionality: 25 (W2V) + 4 (engagement) + 4 (VADER) + 1 (SBERT) = 34",
], note="Show this as a comparison table on the slide."))

# Slide 11 — Results table
story.extend(slide_block(11, "Results", [
    "Test set: 1,145 samples (held-out, never seen during training)",
]))

results_data = [
    ["Model", "Accuracy", "Precision", "Recall", "F1-Score"],
    ["Model 1: Baseline SVM",   "95.81%", "96.45%", "95.10%", "95.77%"],
    ["Model 2: Enhanced SVM ★", "96.16%", "96.32%", "95.98%", "96.15%"],
    ["Model 3: Enhanced RF",    "93.28%", "94.12%", "92.31%", "93.20%"],
]
results_table = Table(results_data, colWidths=[5.5*cm, 2.8*cm, 2.8*cm, 2.5*cm, 2.5*cm])
results_table.setStyle(TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0),  NAVY),
    ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
    ("BACKGROUND",  (0, 2), (-1, 2),  LIGHT),
    ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTNAME",    (0, 2), (-1, 2),  "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, -1), 9),
    ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
    ("ALIGN",       (0, 0), (0, -1),  "LEFT"),
    ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
    ("TOPPADDING",  (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
]))
story.append(results_table)
story.append(Spacer(1, 0.2*cm))
story.extend(bullets([
    "★ Model 2 best overall — +0.35% accuracy and +0.88% recall vs. baseline",
    "Model 2 false negatives: 23 (vs 28 in baseline) — 5 fewer missed clickbaits",
    "Also show confusion matrices for all 3 models on this slide",
]))

# Slide 12
story.extend(slide_block(12, "Discussion / Analysis", [
    "Enhanced SVM improved accuracy (+0.35%) and recall (+0.88%) over baseline",
    "Comment sentiment and title-transcript gap are statistically valid signals ✓",
    "Random Forest underperformed: RBF-SVM handles dense continuous vectors better",
    "Trade-off: small accuracy gain requires significant extra data collection effort",
    "  (API quota management, cookie spoofing, Sentence-BERT encoding overhead)",
]))

# Slide 13
story.extend(slide_block(13, "Demo Video", [
    "Embed a short demo video here (Insert → Video in PowerPoint)",
    "Show: input a YouTube video URL → pipeline runs feature extraction → model outputs CLICKBAIT or NOT CLICKBAIT",
    "Ideal length: 1-2 minutes",
], note="This is its own dedicated slide. Just the video + a minimal title is enough."))

# Slide 14
story.extend(slide_block(14, "Conclusion", [
    "Successfully extended Vierti's SVM baseline with 2 new feature blocks (Block B + C)",
    "Best model: Enhanced SVM at 96.16% accuracy (F1: 96.15%)",
    "Comment sentiment (VADER) and title-transcript mismatch (Sentence-BERT) are effective clickbait signals",
    "Limitation: dataset reduced from ~36,000 to 5,722 due to transcript/comment availability",
    "Limitation: potential selection bias — only videos with public comments and transcripts",
]))

# Slide 15
story.extend(slide_block(15, "Future Work", [
    "Scale up data collection with more API keys and faster extraction pipelines",
    "Add visual features — thumbnail analysis (color, face detection, text overlay)",
    "Try transformer-based classifiers (BERT, RoBERTa) instead of SVM",
    "Deploy as a real-time browser extension for YouTube",
    "Address selection bias by including videos without transcripts (e.g., audio transcription)",
]))

# Slide 16
story.extend(slide_block(16, "Q&A / Thank You", [
    "Team member names",
    "GitHub repository link (if public)",
    "Acknowledgement of advisor: Rhio Sutoyo",
]))

story.append(Spacer(1, 0.5*cm))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: POSTER STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(Paragraph("  PART 2 — ACADEMIC POSTER STRUCTURE  ", section_header))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph(
    "Academic posters are typically A0 or A1 in portrait orientation. "
    "They are read in a Z or column pattern. Below is the recommended layout and content for each section.",
    body
))
story.append(Spacer(1, 0.3*cm))

# Layout diagram via table
layout_data = [
    ["HEADER (full width)\nTitle · Authors · Institution · Course"],
    [""],
]
layout_table = Table(
    [
        [Paragraph("<b>HEADER (full width)</b><br/>Title · Authors · Institution · Course", body)],
        [Table(
            [[
                Paragraph("<b>LEFT COLUMN</b><br/>(~33% width)<br/><br/>• Background<br/>• Problem Statement<br/>• Related Work<br/>• Dataset", body),
                Paragraph("<b>MIDDLE COLUMN</b><br/>(~33% width)<br/><br/>• Methodology / System Pipeline<br/>• Feature Engineering<br/>  (Block A, B, C)", body),
                Paragraph("<b>RIGHT COLUMN</b><br/>(~33% width)<br/><br/>• Results (table + charts)<br/>• Discussion<br/>• Conclusion<br/>• Future Work / References", body),
            ]],
            colWidths=[5.2*cm, 5.2*cm, 5.2*cm],
            style=[
                ("BOX", (0,0), (-1,-1), 1, NAVY),
                ("INNERGRID", (0,0), (-1,-1), 0.5, GRAY),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("TOPPADDING", (0,0), (-1,-1), 8),
                ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                ("LEFTPADDING", (0,0), (-1,-1), 6),
            ]
        )],
    ],
    colWidths=[16.2*cm],
    style=[
        ("BOX", (0,0), (-1,-1), 1.5, NAVY),
        ("BACKGROUND", (0,0), (-1,0), LIGHT),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
    ]
)
story.append(layout_table)
story.append(Spacer(1, 0.4*cm))

# Poster sections
poster_sections = [
    ("Header (Full Width)", [
        "Title of the research (can be shortened from the paper title)",
        "All author names and student IDs",
        "Bina Nusantara University · RMCS · Semester 4 · Year",
        "Optional: University logo or department banner",
    ]),
    ("1. Background & Problem Statement", [
        "2-3 sentences: why clickbait detection matters",
        "What existing approaches miss (no comment sentiment, no title-transcript check)",
        "Research question in 1 clear sentence",
    ]),
    ("2. Related Work", [
        "3-4 key references in brief (1 line each)",
        "Vierti (2019), Gothankar et al. (2022), Elyashar et al. (2022), Bronakowski et al. (2023)",
        "What gap this work fills",
    ]),
    ("3. Dataset", [
        "Source: Vierti's dataset (~36,000 videos)",
        "After filtering: 5,722 balanced videos (2,861 clickbait / 2,861 non-clickbait)",
        "Train / Test: 4,577 / 1,145 — stratified 80/20 split",
        "Include a small data collection pipeline diagram",
    ]),
    ("4. Methodology / System Pipeline", [
        "A clean visual flowchart is essential here — this is the centrepiece of the poster",
        "Show: Dataset → Stage 1 (Extract & Filter) → Block A / Block B / Block C → Concatenate → SVM / RF → Output",
        "Label each block with the key technique (Word2Vec, VADER, Sentence-BERT)",
    ]),
    ("5. Feature Engineering", [
        "Block A: Word2Vec title embeddings (25-dim) + engagement metrics (4-dim) = 29 features",
        "Block B: VADER sentiment on top-100 comments → 4 features (mean, std, %pos, %neg)",
        "Block C: Sentence-BERT cosine similarity (title ↔ transcript) → 1 feature",
        "Total enhanced feature vector: 34 dimensions",
        "Use icons or small diagrams — avoid paragraph text",
    ]),
    ("6. Results", [
        "Comparison table (Model 1 vs 2 vs 3): Accuracy, Precision, Recall, F1",
        "Bar chart or grouped bar chart is more visually impactful than a table alone",
        "Highlight the best result (Model 2: 96.16% accuracy, 96.15% F1)",
        "Include at least one confusion matrix (preferably Model 2)",
        "Key callout stat: '+0.35% accuracy and 5 fewer missed clickbaits'",
    ]),
    ("7. Discussion", [
        "Why Model 2 beat baseline: comment sentiment + semantic gap are valid signals",
        "Why Random Forest underperformed: RBF-SVM handles dense continuous vectors better",
        "Limitation: dataset shrunk significantly due to transcript/comment availability",
        "1-2 bullet points maximum — keep it tight on a poster",
    ]),
    ("8. Conclusion", [
        "Best model: Enhanced SVM, 96.16% accuracy",
        "Comment sentiment (VADER) and title-transcript mismatch (SBERT) are effective features",
        "Open-source reproducible pipeline",
        "2-3 bullets only",
    ]),
    ("9. Future Work & References", [
        "Future: thumbnail analysis, transformer classifiers (BERT/RoBERTa), browser extension",
        "References: 4-5 key citations in compact format (IEEE or APA)",
        "QR code linking to GitHub repo (optional but looks great on a poster)",
    ]),
]

for title_text, items in poster_sections:
    story.append(Paragraph(f"  {title_text}  ", poster_section))
    story.append(Spacer(1, 0.1*cm))
    story.extend(bullets(items))
    story.append(Spacer(1, 0.1*cm))

# Poster tips
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("  POSTER DESIGN TIPS  ", section_header))
story.append(Spacer(1, 0.2*cm))
story.extend(bullets([
    "Size: A0 (84 × 119 cm) or A1 (59 × 84 cm) portrait is standard",
    "Font size: Title ≥ 72pt · Section headers ≥ 36pt · Body text ≥ 24pt (readable from 1 metre away)",
    "60% of the poster should be visuals (diagrams, charts, tables) — not text",
    "Use a consistent 2 or 3 colour palette — stick to it throughout",
    "Every section should have at least one visual element (no text-only blocks)",
    "Keep body text to bullet points — remove full sentences wherever possible",
    "Leave white space — a crowded poster is harder to read than a sparse one",
    "The pipeline/system diagram is the most important visual — make it large and central",
    "QR code in the bottom corner linking to your GitHub or paper is a nice touch",
]))

story.append(Spacer(1, 1*cm))
story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=8))
story.append(Paragraph(
    "Generated for RMCS Final Project · Bina Nusantara University · 2025",
    ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=GRAY, alignment=TA_CENTER)
))

doc.build(story)
print(f"PDF saved: {OUTPUT}")
