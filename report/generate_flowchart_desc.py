from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

OUTPUT = "flowchart_deskripsi_id.pdf"

W, H = A4
MARGIN = 18 * mm

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=MARGIN,
    rightMargin=MARGIN,
    topMargin=16 * mm,
    bottomMargin=16 * mm,
)

# ── Colours ──────────────────────────────────────────────────────────────────
C_DARK   = colors.HexColor("#1a1a2e")
C_BLUE   = colors.HexColor("#1565C0")
C_LBLUE  = colors.HexColor("#E3F2FD")
C_GREEN  = colors.HexColor("#1B5E20")
C_LGREEN = colors.HexColor("#E8F5E9")
C_ORG    = colors.HexColor("#E65100")
C_LORG   = colors.HexColor("#FFF3E0")
C_PURP   = colors.HexColor("#4A148C")
C_LPURP  = colors.HexColor("#F3E5F5")
C_GREY   = colors.HexColor("#F5F5F5")
C_MID    = colors.HexColor("#757575")
C_LINE   = colors.HexColor("#BDBDBD")

# ── Styles ───────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=base[parent], **kw)

sTitle   = S("sTitle",   "Title",  fontSize=16, textColor=C_DARK,
             spaceAfter=2, alignment=TA_CENTER)
sSub     = S("sSub",    "Normal", fontSize=9,  textColor=C_MID,
             alignment=TA_CENTER, spaceAfter=8)
sPhase   = S("sPhase",  "Normal", fontSize=11, textColor=colors.white,
             alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=0, leading=14)
sLabel   = S("sLabel",  "Normal", fontSize=9,  textColor=C_DARK,
             fontName="Helvetica-Bold", spaceAfter=2, leading=12)
sBody    = S("sBody",   "Normal", fontSize=8,  textColor=C_DARK,
             spaceAfter=2, leading=11)
sArrow   = S("sArrow",  "Normal", fontSize=12, textColor=C_MID,
             alignment=TA_CENTER, spaceAfter=0)
sNote    = S("sNote",   "Normal", fontSize=7.5, textColor=C_MID,
             alignment=TA_CENTER, leading=10)
sCaption = S("sCaption","Normal", fontSize=8,  textColor=C_MID,
             alignment=TA_CENTER, spaceAfter=4)
sCanvaH  = S("sCanvaH", "Normal", fontSize=10, textColor=C_DARK,
             fontName="Helvetica-Bold", spaceAfter=3, leading=13)
sCanvaB  = S("sCanvaB", "Normal", fontSize=8,  textColor=C_DARK,
             spaceAfter=2, leading=11)

def arrow():
    return Paragraph("&#8595;", sArrow)

def phase_header(text, bg):
    t = Table([[Paragraph(text, sPhase)]], colWidths=[W - 2*MARGIN])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("ROUNDEDCORNERS", [6,6,6,6]),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]))
    return t

def box(label, bullets, bg, border):
    content = [Paragraph(label, sLabel)]
    for b in bullets:
        content.append(Paragraph(f"• {b}", sBody))
    t = Table([[content]], colWidths=["100%"])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), bg),
        ("BOX",           (0,0), (-1,-1), 1, border),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    return t

def two_col(left_label, left_bullets, right_label, right_bullets,
            bg, border, note=None):
    cw = (W - 2*MARGIN - 4*mm) / 2

    def cell(label, bullets):
        parts = [Paragraph(label, sLabel)]
        for b in bullets:
            parts.append(Paragraph(f"• {b}", sBody))
        return parts

    rows = [[cell(left_label, left_bullets), cell(right_label, right_bullets)]]
    if note:
        rows.append([Paragraph(note, sNote), ""])

    t = Table(rows, colWidths=[cw, cw])
    ts = [
        ("BACKGROUND",   (0,0), (-1,-1), bg),
        ("BOX",          (0,0), (0,0),   1, border),
        ("BOX",          (1,0), (1,0),   1, border),
        ("LINEBEFORE",   (1,0), (1,-1),  0.5, border),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
    ]
    if note:
        ts += [
            ("SPAN",         (0,1), (1,1)),
            ("BACKGROUND",   (0,1), (1,1), colors.white),
            ("TOPPADDING",   (0,1), (1,1), 3),
            ("BOTTOMPADDING",(0,1), (1,1), 3),
        ]
    t.setStyle(TableStyle(ts))
    return t

def three_col(items, bg, border):
    """items = list of (label, bullets) tuples, len 3"""
    cw = (W - 2*MARGIN - 4*mm) / 3

    def cell(label, bullets):
        parts = [Paragraph(label, sLabel)]
        for b in bullets:
            parts.append(Paragraph(f"• {b}", sBody))
        return parts

    row = [cell(l, b) for l, b in items]
    t = Table([row], colWidths=[cw, cw, cw])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), bg),
        ("BOX",          (0,0), (0,0),   1, border),
        ("BOX",          (1,0), (1,0),   1, border),
        ("BOX",          (2,0), (2,0),   1, border),
        ("LINEBEFORE",   (1,0), (1,0),   0.5, border),
        ("LINEBEFORE",   (2,0), (2,0),   0.5, border),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("LEFTPADDING",  (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
    ]))
    return t

def sp(h=4):
    return Spacer(1, h * mm)

# ── Build story ───────────────────────────────────────────────────────────────
story = []

story.append(Paragraph("Alur Metodologi Penelitian", sTitle))
story.append(Paragraph(
    "Deteksi Clickbait pada Video YouTube menggunakan SVM, VADER, dan SBERT", sSub))
story.append(HRFlowable(width="100%", thickness=1, color=C_LINE))
story.append(sp(4))

# ── FASE 1 ───────────────────────────────────────────────────────────────────
story.append(phase_header("FASE 1 — Pengumpulan Data", C_BLUE))
story.append(sp(2))
story.append(Paragraph(
    "Dua sumber data dikumpulkan secara paralel, lalu digabungkan.", sCaption))
story.append(sp(1))

story.append(two_col(
    "Kolom Kiri: Dataset yang Ada (Vierti et al.)",
    [
        "File CSV sudah tersedia",
        "Berisi: judul video, views, likes, dislikes, jumlah komentar",
        "Label sudah ada: clickbait (1) / non-clickbait (0)",
    ],
    "Kolom Kanan: Pengambilan Data YouTube",
    [
        "Ambil ID video dari dataset",
        "Unduh transkrip otomatis via YouTube Transcript API",
        "Unduh komentar video via YouTube API",
    ],
    C_LBLUE, C_BLUE,
    note="Kedua alur di atas digabung di langkah berikutnya  ↓",
))
story.append(sp(2))
story.append(arrow())
story.append(sp(1))
story.append(box(
    "Penggabungan & Penyaringan Data",
    [
        "Cek ketersediaan: simpan hanya video yang PUNYA transkrip DAN komentar",
        "Penyeimbangan kelas: jumlah clickbait = jumlah non-clickbait (downsample)",
        "Hasil akhir: 5.722 video (50% clickbait, 50% non-clickbait)",
    ],
    C_GREY, C_LINE,
))
story.append(sp(3))

# ── FASE 2 ───────────────────────────────────────────────────────────────────
story.append(phase_header("FASE 2 — Ekstraksi Fitur (3 Blok Paralel)", C_GREEN))
story.append(sp(2))
story.append(Paragraph(
    "Tiga blok fitur diekstraksi secara terpisah dari sumber data yang berbeda.", sCaption))
story.append(sp(1))

story.append(three_col([
    (
        "Blok A — Judul & Engagement",
        [
            "Word2Vec pada judul video",
            "→ 25 dimensi vektor kata",
            "Views, likes, dislikes, komentar",
            "→ log-transform + normalisasi",
            "Total: 29 fitur",
        ],
    ),
    (
        "Blok B — Sentimen Komentar",
        [
            "VADER dijalankan pada setiap komentar",
            "Rata-rata skor compound",
            "Standar deviasi skor",
            "% komentar positif",
            "% komentar negatif",
            "Total: 4 fitur",
        ],
    ),
    (
        "Blok C — Kemiripan Semantik",
        [
            "SBERT mengubah judul & transkrip jadi vektor",
            "Hitung cosine similarity antar keduanya",
            "Tinggi = judul sesuai isi video",
            "Rendah = judul menyesatkan (clickbait)",
            "Total: 1 fitur",
        ],
    ),
], C_LGREEN, C_GREEN))
story.append(sp(2))
story.append(arrow())
story.append(sp(1))
story.append(box(
    "Penggabungan Fitur",
    ["Blok A + Blok B + Blok C digabung menjadi satu vektor fitur: 34 fitur total per video"],
    C_GREY, C_LINE,
))
story.append(sp(3))

# ── FASE 3 ───────────────────────────────────────────────────────────────────
story.append(phase_header("FASE 3 — Pelatihan Model", C_ORG))
story.append(sp(2))
story.append(Paragraph(
    "Data dibagi 80% latih / 20% uji, lalu tiga model dilatih secara terpisah.", sCaption))
story.append(sp(1))

story.append(box(
    "Pembagian Data: 80% Latih  |  20% Uji",
    ["Split dilakukan sekali dan dipakai oleh ketiga model agar perbandingan adil"],
    C_GREY, C_LINE,
))
story.append(sp(2))
story.append(arrow())
story.append(sp(1))

story.append(three_col([
    (
        "Model 1 — Baseline SVM",
        [
            "Fitur: Blok A saja (29 fitur)",
            "Replikasi Vierti et al.",
            "Hyperparameter tetap:",
            "C=3.7, gamma=4.1, kernel RBF",
            "Tujuan: patokan awal",
        ],
    ),
    (
        "Model 2 — Enhanced SVM",
        [
            "Fitur: Blok A+B+C (34 fitur)",
            "Hyperparameter dicari otomatis",
            "5-fold cross-validation",
            "Grid search pada C & gamma",
            "Kernel RBF",
        ],
    ),
    (
        "Model 3 — Enhanced RF",
        [
            "Fitur: Blok A+B+C (34 fitur)",
            "Hyperparameter dicari otomatis",
            "5-fold cross-validation",
            "Grid search: n_estimators,",
            "max_depth, min_samples_split",
        ],
    ),
], C_LORG, C_ORG))
story.append(sp(3))

# ── FASE 4 ───────────────────────────────────────────────────────────────────
story.append(phase_header("FASE 4 — Evaluasi", C_PURP))
story.append(sp(2))
story.append(box(
    "Metrik Evaluasi (diukur pada data uji 20%)",
    [
        "Akurasi — seberapa sering prediksi benar secara keseluruhan",
        "Presisi — dari yang diprediksi clickbait, berapa yang benar-benar clickbait",
        "Recall — dari semua clickbait asli, berapa yang berhasil terdeteksi",
        "F1-Score — rata-rata harmonis presisi dan recall (metrik utama)",
        "Ketiga model dibandingkan berdampingan pada tabel yang sama",
    ],
    C_LPURP, C_PURP,
))
story.append(sp(4))

# ── PANDUAN CANVA ────────────────────────────────────────────────────────────
story.append(HRFlowable(width="100%", thickness=1, color=C_LINE))
story.append(sp(3))
story.append(Paragraph("Panduan Layout di Canva", sCanvaH))
story.append(sp(1))

tips = [
    ("Ukuran kanvas",
     "Gunakan ukuran A4 (210 × 297 mm) agar pas di laporan. "
     "Di Canva: klik 'Buat desain' → ketik 'A4 Document'."),
    ("Struktur dua kolom (Fase 1)",
     "Buat satu kotak lebar di atas sebagai header fase. "
     "Di bawahnya, letakkan dua kotak berdampingan (kiri & kanan). "
     "Hubungkan keduanya ke satu kotak di bawah dengan dua panah yang mengarah ke bawah-tengah."),
    ("Struktur tiga kolom (Fase 2 & 3)",
     "Sama seperti dua kolom, tapi tiga kotak sejajar. "
     "Setelah ketiga kotak, tarik tiga panah menuju satu kotak 'penggabungan' di tengah bawah."),
    ("Kotak dengan garis putus-putus",
     "Tambahkan persegi panjang → klik 'Border' → pilih garis putus-putus, "
     "isi transparan. Gunakan ini sebagai pembatas tiap fase."),
    ("Membuat kotak sama besar",
     "Pilih semua kotak yang ingin disamakan → klik kanan → 'Samakan ukuran'. "
     "Lalu gunakan 'Atur → Ratakan jarak' agar seragam."),
    ("Panah penghubung",
     "Gunakan Elements → Lines → Arrow. "
     "Tarik dari ujung bawah satu kotak ke ujung atas kotak berikutnya. "
     "Untuk dua panah menyatu ke satu kotak: tarik dua panah terpisah ke kotak yang sama."),
    ("Warna per fase (saran)",
     "Fase 1: Biru | Fase 2: Hijau | Fase 3: Oranye | Fase 4: Ungu. "
     "Konsisten dengan warna yang sama memudahkan pembaca mengikuti alur."),
]

for title, desc in tips:
    story.append(Paragraph(f"<b>{title}</b>", sCanvaB))
    story.append(Paragraph(desc, sCanvaB))
    story.append(sp(1.5))

doc.build(story)
print(f"PDF saved: {OUTPUT}")
