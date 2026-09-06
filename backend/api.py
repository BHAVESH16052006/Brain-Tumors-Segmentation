import os
import uuid
import io
import numpy as np
import SimpleITK as sitk

from pathlib import Path
from PIL import Image

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak,
)


from predict import load_model, predict_patient


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\chand\BrainTumorProject\Brain-Tumors-Segmentation"
)

UPLOAD_ROOT = PROJECT_DIR / "user_predictions" / "uploads"
PREDICTION_ROOT = PROJECT_DIR / "user_predictions"

UPLOAD_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Brain Tumor Segmentation API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL MODEL
# ============================================================

MODEL = None


# ============================================================
# CURRENT CASE CACHE
#
# Keeps the latest uploaded case in memory so the frontend
# can request MRI slices without running inference again.
# ============================================================

CASES = {}


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    global MODEL

    print()
    print("=" * 60)
    print("Loading Brain Tumor Segmentation Model")
    print("=" * 60)

    MODEL = load_model()

    print("Model loaded successfully.")
    print("=" * 60)
    print()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "message": "Brain Tumor Segmentation API is running!"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model_loaded": MODEL is not None
    }


# ============================================================
# HELPERS
# ============================================================

def calculate_volume_cm3(
    voxel_count,
    spacing
):
    """
    Calculate physical volume from voxel count.

    spacing is in mm:
        (x, y, z)

    One cm³ = 1000 mm³.
    """

    voxel_volume_mm3 = (
        float(spacing[0]) *
        float(spacing[1]) *
        float(spacing[2])
    )

    volume_cm3 = (
        float(voxel_count) *
        voxel_volume_mm3 /
        1000.0
    )

    return volume_cm3


def get_tumor_burden_category(volume_cm3):
    """Research/demo tumor-burden category based on segmented volume."""
    volume_cm3 = float(volume_cm3)
    if volume_cm3 <= 0:
        return "No Tumor"
    if volume_cm3 < 20:
        return "Low"
    if volume_cm3 < 50:
        return "Moderate"
    if volume_cm3 < 100:
        return "High"
    return "Very High"


def normalize_for_display(image):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    finite_values = image[
        np.isfinite(image)
    ]

    if finite_values.size == 0:
        return np.zeros(
            image.shape,
            dtype=np.uint8
        )

    low = np.percentile(
        finite_values,
        1
    )

    high = np.percentile(
        finite_values,
        99
    )

    if high <= low:
        high = low + 1.0

    image = np.clip(
        image,
        low,
        high
    )

    image = (
        (image - low) /
        (high - low) *
        255.0
    )

    return image.astype(
        np.uint8
    )


def make_mri_png(
    array,
    slice_index
):
    """
    Convert a 3D [Z,Y,X] MRI volume slice into PNG.
    """

    total_slices = array.shape[0]

    slice_index = max(
        0,
        min(
            int(slice_index),
            total_slices - 1
        )
    )

    image_slice = array[
        slice_index,
        :,
        :
    ]

    image_slice = normalize_for_display(
        image_slice
    )

    image = Image.fromarray(
        image_slice,
        mode="L"
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


def make_overlay_png(
    mri_array,
    prediction,
    slice_index
):
    """
    Create MRI + segmentation overlay.

    Labels:
        1 = NCR / NET
        2 = Edema
        3 = Enhancing Tumor
    """

    total_slices = mri_array.shape[0]

    slice_index = max(
        0,
        min(
            int(slice_index),
            total_slices - 1
        )
    )

    mri_slice = mri_array[
        slice_index,
        :,
        :
    ]

    pred_slice = prediction[
        slice_index,
        :,
        :
    ]

    gray = normalize_for_display(
        mri_slice
    )

    # RGB grayscale image
    rgb = np.stack(
        [gray, gray, gray],
        axis=-1
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Segmentation colors
    #
    # Edema           -> green
    # NCR / NET       -> pink/red
    # Enhancing Tumor -> purple
    # --------------------------------------------------------

    edema_color = np.array(
        [54, 211, 153],
        dtype=np.float32
    )

    ncr_color = np.array(
        [255, 92, 120],
        dtype=np.float32
    )

    enhancing_color = np.array(
        [188, 92, 255],
        dtype=np.float32
    )

    alpha = 0.55

    # Edema = label 2
    mask = pred_slice == 2

    rgb[mask] = (
        rgb[mask] * (1.0 - alpha) +
        edema_color * alpha
    )

    # NCR / NET = label 1
    mask = pred_slice == 1

    rgb[mask] = (
        rgb[mask] * (1.0 - alpha) +
        ncr_color * alpha
    )

    # Enhancing tumor = label 3
    mask = pred_slice == 3

    rgb[mask] = (
        rgb[mask] * (1.0 - alpha) +
        enhancing_color * alpha
    )

    rgb = np.clip(
        rgb,
        0,
        255
    ).astype(
        np.uint8
    )

    image = Image.fromarray(
        rgb,
        mode="RGB"
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


def find_prediction_file(
    case_id
):
    """
    Find prediction NIfTI generated for this case.
    """

    candidates = [
        PREDICTION_ROOT /
        f"{case_id}-prediction.nii.gz",

        PREDICTION_ROOT /
        f"{case_id}_prediction.nii.gz",
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    # Search recursively as a fallback.
    matches = list(
        PREDICTION_ROOT.rglob(
            "*prediction.nii.gz"
        )
    )

    if matches:
        # Most recent prediction
        matches.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        return matches[0]

    return None



# ============================================================
# PDF REPORT
# ============================================================

def make_pdf_report(
    case_id,
    case,
    result,
    selected_slice=None
):
    """
    Generate a professional 3-page PDF report.

    Page 1:
        Analysis summary and physical measurements.

    Page 2:
        T1 Native, T1 Contrast, T2 Weighted, T2 FLAIR.

    Page 3:
        Segmentation overlay and legend.

    The same display functions used by the web viewer are used
    here, so the PDF stays aligned with the UI.
    """

    report_dir = PREDICTION_ROOT / "reports"

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        report_dir /
        f"{case_id}-brain-tumor-report.pdf"
    )

    spacing = case["spacing"]
    size = case["size"]
    num_slices = int(
        case["num_slices"]
    )

    prediction = np.asarray(
        case["prediction"]
    )

    # --------------------------------------------------------
    # Measurements
    # --------------------------------------------------------

    tumor_voxels = int(
        np.sum(
            prediction > 0
        )
    )

    edema_voxels = int(
        np.sum(
            prediction == 2
        )
    )

    ncr_net_voxels = int(
        np.sum(
            prediction == 1
        )
    )

    enhancing_tumor_voxels = int(
        np.sum(
            prediction == 3
        )
    )

    tumor_volume_cm3 = calculate_volume_cm3(
        tumor_voxels,
        spacing
    )

    edema_volume_cm3 = calculate_volume_cm3(
        edema_voxels,
        spacing
    )

    ncr_net_volume_cm3 = calculate_volume_cm3(
        ncr_net_voxels,
        spacing
    )

    enhancing_tumor_volume_cm3 = calculate_volume_cm3(
        enhancing_tumor_voxels,
        spacing
    )

    voxel_volume_mm3 = (
        float(spacing[0]) *
        float(spacing[1]) *
        float(spacing[2])
    )

    tumor_detected = (
        tumor_voxels > 0
    )

    detection_text = (
        "TUMOR DETECTED"
        if tumor_detected
        else "NO TUMOR DETECTED"
    )

    # --------------------------------------------------------
    # Selected slice
    # --------------------------------------------------------

    if selected_slice is None:

        tumor_slice_counts = np.sum(
            prediction > 0,
            axis=(1, 2)
        )

        if np.max(
            tumor_slice_counts
        ) > 0:

            selected_slice = int(
                np.argmax(
                    tumor_slice_counts
                )
            )

        else:

            selected_slice = (
                num_slices // 2
            )

    selected_slice = max(
        0,
        min(
            int(selected_slice),
            num_slices - 1
        )
    )

    # --------------------------------------------------------
    # Temporary image directory
    # --------------------------------------------------------

    import shutil
    import tempfile

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{case_id}_report_"
        )
    )

    try:

        modalities = [
            ("t1n", "T1 Native"),
            ("t1c", "T1 Contrast"),
            ("t2w", "T2 Weighted"),
            ("t2f", "T2 FLAIR"),
        ]

        image_paths = []

        # ----------------------------------------------------
        # Generate four MRI PNGs.
        # ----------------------------------------------------

        for modality, title in modalities:

            image = sitk.ReadImage(
                str(
                    case["files"][modality]
                )
            )

            array = sitk.GetArrayFromImage(
                image
            )

            png_data = make_mri_png(
                array,
                selected_slice
            )

            image_path = (
                temp_dir /
                f"{modality}.png"
            )

            image_path.write_bytes(
                png_data
            )

            image_paths.append(
                (
                    title,
                    image_path
                )
            )

        # ----------------------------------------------------
        # Generate T1 Native segmentation overlay.
        # ----------------------------------------------------

        t1n_image = sitk.ReadImage(
            str(
                case["files"]["t1n"]
            )
        )

        t1n_array = sitk.GetArrayFromImage(
            t1n_image
        )

        overlay_png = make_overlay_png(
            t1n_array,
            prediction,
            selected_slice
        )

        overlay_path = (
            temp_dir /
            "overlay.png"
        )

        overlay_path.write_bytes(
            overlay_png
        )

        # ----------------------------------------------------
        # PDF document
        # ----------------------------------------------------

        def draw_report_page(canvas_obj, doc_obj):
            canvas_obj.saveState()

            page_width, page_height = A4

            # Header line
            canvas_obj.setStrokeColor(colors.HexColor("#D9E2EC"))
            canvas_obj.setLineWidth(0.6)
            canvas_obj.line(15 * mm, page_height - 10 * mm, page_width - 15 * mm, page_height - 10 * mm)

            canvas_obj.setFont("Helvetica-Bold", 8)
            canvas_obj.setFillColor(colors.HexColor("#26364A"))
            canvas_obj.drawString(15 * mm, page_height - 7.5 * mm, "NEUROSCAN AI")

            canvas_obj.setFont("Helvetica", 7.5)
            canvas_obj.setFillColor(colors.HexColor("#667085"))
            canvas_obj.drawRightString(page_width - 15 * mm, page_height - 7.5 * mm, "Brain Tumor Segmentation Report")

            # Footer
            canvas_obj.setStrokeColor(colors.HexColor("#E5E7EB"))
            canvas_obj.line(15 * mm, 10 * mm, page_width - 15 * mm, 10 * mm)
            canvas_obj.setFont("Helvetica", 7)
            canvas_obj.setFillColor(colors.HexColor("#667085"))
            canvas_obj.drawString(15 * mm, 6.5 * mm, f"Case ID: {case_id}")
            canvas_obj.drawCentredString(page_width / 2, 6.5 * mm, "AI-assisted segmentation • Research / educational use")
            canvas_obj.drawRightString(page_width - 15 * mm, 6.5 * mm, f"Page {doc_obj.page}")

            canvas_obj.restoreState()

        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=A4,
            rightMargin=15 * mm,
            leftMargin=15 * mm,
            topMargin=18 * mm,
            bottomMargin=15 * mm,
            title="NeuroScan AI Brain Tumor Segmentation Report",
            author="NeuroScan AI",
            subject="Multimodal brain MRI tumor segmentation analysis",
            creator="NeuroScan AI",
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "NeuroScanTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=23,
            leading=27,
            spaceAfter=6,
        )

        subtitle_style = ParagraphStyle(
            "NeuroScanSubtitle",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            fontSize=10,
            leading=14,
            spaceAfter=16,
        )

        heading_style = ParagraphStyle(
            "ReportHeading",
            parent=styles["Heading2"],
            fontSize=15,
            leading=19,
            spaceBefore=5,
            spaceAfter=9,
        )

        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
        )

        result_style = ParagraphStyle(
            "ReportResult",
            parent=styles["Heading1"],
            alignment=TA_CENTER,
            fontSize=18,
            leading=22,
            spaceBefore=8,
            spaceAfter=14,
        )

        small_style = ParagraphStyle(
            "ReportSmall",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
        )

        story = []

        # ====================================================
        # PAGE 1 — ANALYSIS SUMMARY
        # ====================================================

        story.append(
            Paragraph(
                "NeuroScan AI",
                title_style
            )
        )

        story.append(
            Paragraph(
                "Brain Tumor Segmentation Report",
                subtitle_style
            )
        )

        story.append(
            Paragraph(
                f"<b>Case ID:</b> {case_id}",
                body_style
            )
        )

        story.append(
            Spacer(
                1,
                7
            )
        )

        story.append(
            Paragraph(
                detection_text,
                result_style
            )
        )

        summary_data = [
            [
                "Measurement",
                "Value"
            ],
            [
                "Total Tumor Volume",
                f"{tumor_volume_cm3:.2f} cm³"
            ],
            [
                "Edema Volume",
                f"{edema_volume_cm3:.2f} cm³"
            ],
            [
                "NCR / NET Volume",
                f"{ncr_net_volume_cm3:.2f} cm³"
            ],
            [
                "Enhancing Tumor Volume",
                f"{enhancing_tumor_volume_cm3:.2f} cm³"
            ],
        ]

        summary_table = Table(
            summary_data,
            colWidths=[
                95 * mm,
                65 * mm
            ]
        )

        summary_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#202938"
                    )
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#CCCCCC"
                    )
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
            ])
        )

        story.append(
            summary_table
        )

        story.append(
            Spacer(
                1,
                18
            )
        )

        story.append(
            Paragraph(
                "Technical Information",
                heading_style
            )
        )

        technical_data = [
            [
                "Parameter",
                "Value"
            ],
            [
                "Voxel Spacing",
                (
                    f"{float(spacing[0]):.4f} × "
                    f"{float(spacing[1]):.4f} × "
                    f"{float(spacing[2]):.4f} mm"
                )
            ],
            [
                "Voxel Volume",
                f"{voxel_volume_mm3:.4f} mm³"
            ],
            [
                "MRI Dimensions",
                (
                    f"{int(size[0])} × "
                    f"{int(size[1])} × "
                    f"{num_slices}"
                )
            ],
            [
                "Report Slice",
                (
                    f"{selected_slice + 1} / "
                    f"{num_slices}"
                )
            ],
            [
                "Model",
                "SegResNet"
            ],
        ]

        technical_table = Table(
            technical_data,
            colWidths=[
                55 * mm,
                105 * mm
            ]
        )

        technical_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#202938"
                    )
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#CCCCCC"
                    )
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
            ])
        )

        story.append(
            technical_table
        )

        story.append(
            Spacer(
                1,
                16
            )
        )

        story.append(
            Paragraph(
                "Generated: "
                + __import__(
                    "datetime"
                ).datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                small_style
            )
        )

        story.append(
            PageBreak()
        )

        # ====================================================
        # PAGE 2 — MRI VISUALIZATION
        # ====================================================

        story.append(
            Paragraph(
                "MRI Visualization",
                heading_style
            )
        )

        story.append(
            Paragraph(
                (
                    f"Displayed slice: "
                    f"{selected_slice + 1} / "
                    f"{num_slices}"
                ),
                body_style
            )
        )

        story.append(
            Spacer(
                1,
                9
            )
        )

        # Two images per row.
        for row_start in range(
            0,
            len(image_paths),
            2
        ):

            row_images = []
            row_labels = []

            for offset in range(2):

                index = (
                    row_start +
                    offset
                )

                if index < len(
                    image_paths
                ):

                    title, path = (
                        image_paths[index]
                    )

                    row_images.append(
                        RLImage(
                            str(path),
                            width=70 * mm,
                            height=70 * mm,
                        )
                    )

                    row_labels.append(
                        Paragraph(
                            f"<b>{title}</b>",
                            body_style
                        )
                    )

                else:

                    row_images.append("")
                    row_labels.append("")

            image_table = Table(
                [row_images],
                colWidths=[
                    80 * mm,
                    80 * mm
                ]
            )

            image_table.setStyle(
                TableStyle([
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER"
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE"
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        3
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        3
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        3
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        3
                    ),
                ])
            )

            story.append(
                image_table
            )

            label_table = Table(
                [row_labels],
                colWidths=[
                    80 * mm,
                    80 * mm
                ]
            )

            label_table.setStyle(
                TableStyle([
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER"
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        8
                    ),
                ])
            )

            story.append(
                label_table
            )

        story.append(
            PageBreak()
        )

        # ====================================================
        # PAGE 3 — SEGMENTATION
        # ====================================================

        story.append(
            Paragraph(
                "Segmentation Overlay",
                heading_style
            )
        )

        story.append(
            Paragraph(
                (
                    "T1 Native MRI with predicted "
                    "tumor segmentation"
                ),
                body_style
            )
        )

        story.append(
            Spacer(
                1,
                8
            )
        )

        overlay_image = RLImage(
            str(overlay_path),
            width=120 * mm,
            height=120 * mm,
        )

        overlay_table = Table(
            [[overlay_image]],
            colWidths=[
                160 * mm
            ]
        )

        overlay_table.setStyle(
            TableStyle([
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
            ])
        )

        story.append(
            overlay_table
        )

        story.append(
            Spacer(
                1,
                8
            )
        )

        legend_data = [
            [
                "Color",
                "Segmentation"
            ],
            [
                "Green",
                "Edema"
            ],
            [
                "Pink / Red",
                "NCR / NET"
            ],
            [
                "Purple",
                "Enhancing Tumor"
            ],
        ]

        legend_table = Table(
            legend_data,
            colWidths=[
                55 * mm,
                105 * mm
            ]
        )

        legend_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#202938"
                    )
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#CCCCCC"
                    )
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
            ])
        )

        story.append(
            legend_table
        )

        story.append(
            Spacer(
                1,
                12
            )
        )

        story.append(
            Paragraph(
                "<b>Disclaimer:</b> This report is "
                "generated by an AI-based image "
                "segmentation system and is intended "
                "for research and educational purposes. "
                "It is not a substitute for professional "
                "medical diagnosis.",
                small_style
            )
        )

        doc.build(
            story,
            onFirstPage=draw_report_page,
            onLaterPages=draw_report_page,
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

    return report_path


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(request: Request):

    global MODEL

    if MODEL is None:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Model is not loaded."
            }
        )

    try:

        # ----------------------------------------------------
        # Read multipart form
        # ----------------------------------------------------

        form = await request.form()

        required_modalities = [
            "t1n",
            "t1c",
            "t2w",
            "t2f"
        ]

        for modality in required_modalities:

            if modality not in form:

                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error":
                            f"Missing MRI modality: {modality}"
                    }
                )

        # ----------------------------------------------------
        # Generate case ID
        # ----------------------------------------------------

        case_id = (
            "uploaded_" +
            uuid.uuid4().hex[:8]
        )

        case_dir = (
            UPLOAD_ROOT /
            case_id
        )

        case_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Save uploaded files
        # ----------------------------------------------------

        saved_files = {}

        for modality in required_modalities:

            uploaded_file = form[modality]

            original_name = getattr(
                uploaded_file,
                "filename",
                None
            )

            if not original_name:
                original_name = (
                    f"{modality}.nii.gz"
                )

            extension = (
                ".nii.gz"
                if original_name.lower().endswith(
                    ".nii.gz"
                )
                else ".nii"
            )

            output_file = (
                case_dir /
                f"{case_id}-{modality}{extension}"
            )

            content = await uploaded_file.read()

            with open(
                output_file,
                "wb"
            ) as f:

                f.write(content)

            saved_files[
                modality
            ] = output_file

        # ----------------------------------------------------
        # Read T1N spacing
        #
        # This is used for physical volume calculation.
        # ----------------------------------------------------

        t1n_image = sitk.ReadImage(
            str(
                saved_files["t1n"]
            )
        )

        spacing = t1n_image.GetSpacing()

        size = t1n_image.GetSize()

        t1n_array = sitk.GetArrayFromImage(
            t1n_image
        ).astype(np.float32)

        # ----------------------------------------------------
        # Run existing trained model
        # ----------------------------------------------------

        prediction_result = predict_patient(
            str(case_dir),
            MODEL
        )

        # ----------------------------------------------------
        # Extract prediction
        # ----------------------------------------------------

        prediction = np.asarray(
            prediction_result[
                "prediction"
            ]
        )

        # ----------------------------------------------------
        # Extract voxel counts
        # ----------------------------------------------------

        tumor_voxels = int(
            prediction_result.get(
                "tumor_voxels",
                np.sum(
                    prediction > 0
                )
            )
        )

        edema_voxels = int(
            prediction_result.get(
                "edema_voxels",
                np.sum(
                    prediction == 2
                )
            )
        )

        ncr_net_voxels = int(
            prediction_result.get(
                "ncr_net_voxels",
                np.sum(
                    prediction == 1
                )
            )
        )

        enhancing_tumor_voxels = int(
            prediction_result.get(
                "enhancing_tumor_voxels",
                np.sum(
                    prediction == 3
                )
            )
        )

        # ----------------------------------------------------
        # Physical voxel volume
        # ----------------------------------------------------

        voxel_volume_mm3 = (
            float(spacing[0]) *
            float(spacing[1]) *
            float(spacing[2])
        )

        # ----------------------------------------------------
        # Physical volumes in cm³
        # ----------------------------------------------------

        tumor_volume_cm3 = (
            calculate_volume_cm3(
                tumor_voxels,
                spacing
            )
        )

        edema_volume_cm3 = (
            calculate_volume_cm3(
                edema_voxels,
                spacing
            )
        )

        ncr_net_volume_cm3 = (
            calculate_volume_cm3(
                ncr_net_voxels,
                spacing
            )
        )

        enhancing_tumor_volume_cm3 = (
            calculate_volume_cm3(
                enhancing_tumor_voxels,
                spacing
            )
        )

        # ----------------------------------------------------
        # Feature 1 — tumor composition percentages
        # ----------------------------------------------------

        if tumor_voxels > 0:
            edema_percentage = edema_voxels / tumor_voxels * 100.0
            ncr_net_percentage = ncr_net_voxels / tumor_voxels * 100.0
            enhancing_tumor_percentage = enhancing_tumor_voxels / tumor_voxels * 100.0
        else:
            edema_percentage = 0.0
            ncr_net_percentage = 0.0
            enhancing_tumor_percentage = 0.0

        # ----------------------------------------------------
        # Feature 2 — model-derived segmentation confidence
        # ----------------------------------------------------

        tumor_confidence = float(
            prediction_result.get(
                "tumor_confidence",
                0.0
            )
        )
        tumor_confidence = max(0.0, min(100.0, tumor_confidence))

        if tumor_confidence >= 85:
            confidence_level = "HIGH"
        elif tumor_confidence >= 70:
            confidence_level = "MODERATE"
        else:
            confidence_level = "LOW"

        # ----------------------------------------------------
        # Feature 3 — slice-wise tumor burden heatmap data
        # ----------------------------------------------------

        slice_tumor_voxels = np.sum(
            prediction > 0,
            axis=(1, 2)
        ).astype(np.int32)

        slice_brain_voxels = np.sum(
            np.abs(t1n_array) > 0,
            axis=(1, 2)
        ).astype(np.int32)

        slice_burden_percent = np.divide(
            slice_tumor_voxels * 100.0,
            slice_brain_voxels,
            out=np.zeros(slice_tumor_voxels.shape, dtype=np.float32),
            where=slice_brain_voxels > 0
        )

        max_burden_slice = int(np.argmax(slice_tumor_voxels)) if slice_tumor_voxels.size else 0
        max_slice_burden_percent = float(slice_burden_percent[max_burden_slice]) if slice_burden_percent.size else 0.0

        # ----------------------------------------------------
        # Feature 4 — research tumor-burden category
        # ----------------------------------------------------

        tumor_burden_category = get_tumor_burden_category(
            tumor_volume_cm3
        )

        # ----------------------------------------------------
        # Number of slices
        # ----------------------------------------------------

        num_slices = int(
            t1n_image.GetSize()[2]
        )

        # ----------------------------------------------------
        # Determine initial slice
        # ----------------------------------------------------

        tumor_slice_counts = np.sum(
            prediction > 0,
            axis=(1, 2)
        )

        if np.max(
            tumor_slice_counts
        ) > 0:

            initial_slice = int(
                np.argmax(
                    tumor_slice_counts
                )
            )

        else:

            initial_slice = (
                num_slices // 2
            )

        # ----------------------------------------------------
        # Store case information
        # ----------------------------------------------------

        CASES[case_id] = {
            "case_dir": case_dir,
            "files": saved_files,
            "prediction": prediction,
            "spacing": spacing,
            "size": size,
            "num_slices": num_slices,
            "initial_slice": initial_slice,
            "tumor_confidence": tumor_confidence,
            "confidence_level": confidence_level,
            "edema_percentage": edema_percentage,
            "ncr_net_percentage": ncr_net_percentage,
            "enhancing_tumor_percentage": enhancing_tumor_percentage,
            "slice_tumor_voxels": slice_tumor_voxels.tolist(),
            "slice_burden_percent": slice_burden_percent.tolist(),
            "max_burden_slice": max_burden_slice,
            "max_slice_burden_percent": max_slice_burden_percent,
            "tumor_burden_category": tumor_burden_category
        }

        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return {
            "success": True,

            "case_id": case_id,

            "patient_id": case_id,

            "result":
                prediction_result.get(
                    "result",
                    "TUMOR DETECTED"
                    if tumor_voxels > 0
                    else "NO TUMOR DETECTED"
                ),

            "tumor_detected":
                tumor_voxels > 0,

            # -----------------------------------------------
            # Voxel counts
            # -----------------------------------------------

            "tumor_voxels":
                tumor_voxels,

            "edema_voxels":
                edema_voxels,

            "ncr_net_voxels":
                ncr_net_voxels,

            "enhancing_tumor_voxels":
                enhancing_tumor_voxels,

            # -----------------------------------------------
            # Physical volumes
            # -----------------------------------------------

            "tumor_volume_cm3":
                round(
                    tumor_volume_cm3,
                    2
                ),

            "edema_volume_cm3":
                round(
                    edema_volume_cm3,
                    2
                ),

            "ncr_net_volume_cm3":
                round(
                    ncr_net_volume_cm3,
                    2
                ),

            "enhancing_tumor_volume_cm3":
                round(
                    enhancing_tumor_volume_cm3,
                    2
                ),

            "edema_percentage": round(edema_percentage, 2),
            "ncr_net_percentage": round(ncr_net_percentage, 2),
            "enhancing_tumor_percentage": round(enhancing_tumor_percentage, 2),
            "tumor_confidence": round(tumor_confidence, 2),
            "confidence_level": confidence_level,
            "slice_tumor_voxels": slice_tumor_voxels.tolist(),
            "slice_burden_percent": [round(float(v), 2) for v in slice_burden_percent],
            "max_burden_slice": max_burden_slice,
            "max_slice_burden_percent": round(max_slice_burden_percent, 2),
            "tumor_burden_category": tumor_burden_category,

            # -----------------------------------------------
            # Technical information
            # -----------------------------------------------

            "voxel_spacing_mm": [
                float(spacing[0]),
                float(spacing[1]),
                float(spacing[2])
            ],

            "voxel_volume_mm3":
                round(
                    voxel_volume_mm3,
                    4
                ),

            "dimensions": {
                "x": int(size[0]),
                "y": int(size[1]),
                "slices": num_slices
            },

            "num_slices":
                num_slices,

            "initial_slice":
                initial_slice
        }

    except Exception as e:

        import traceback

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


# ============================================================
# MRI SLICE
# ============================================================

@app.get("/case/{case_id}/slice")
async def get_slice(
    case_id: str,
    request: Request
):

    try:

        if case_id not in CASES:

            return JSONResponse(
                status_code=404,
                content={
                    "error":
                        "Case not found. Please run analysis again."
                }
            )

        case = CASES[
            case_id
        ]

        params = request.query_params

        modality = params.get(
            "modality",
            "t1n"
        )

        slice_value = params.get(
            "slice",
            params.get(
                "slice_index",
                "0"
            )
        )

        overlay_value = params.get(
            "overlay",
            "false"
        )

        slice_index = int(
            slice_value
        )

        overlay = (
            overlay_value.lower()
            in [
                "true",
                "1",
                "yes"
            ]
        )

        if modality not in [
            "t1n",
            "t1c",
            "t2w",
            "t2f"
        ]:

            return JSONResponse(
                status_code=400,
                content={
                    "error":
                        "Invalid MRI modality."
                }
            )

        file_path = (
            case["files"][
                modality
            ]
        )

        image = sitk.ReadImage(
            str(file_path)
        )

        array = sitk.GetArrayFromImage(
            image
        )

        slice_index = max(
            0,
            min(
                slice_index,
                array.shape[0] - 1
            )
        )

        if overlay:

            png_data = make_overlay_png(
                array,
                case["prediction"],
                slice_index
            )

        else:

            png_data = make_mri_png(
                array,
                slice_index
            )

        return Response(
            content=png_data,
            media_type="image/png",
            headers={
                "Cache-Control":
                    "no-store"
            }
        )

    except Exception as e:

        import traceback

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )



# ============================================================
# PDF REPORT ENDPOINT
# ============================================================

@app.get("/case/{case_id}/report")
async def generate_report(
    case_id: str,
    request: Request
):

    try:

        if case_id not in CASES:

            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error":
                        "Case not found. Please run analysis again."
                }
            )

        case = CASES[
            case_id
        ]

        slice_value = request.query_params.get(
            "slice",
            str(
                case.get(
                    "initial_slice",
                    case["num_slices"] // 2
                )
            )
        )

        try:
            selected_slice = int(
                slice_value
            )
        except ValueError:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error":
                        "Invalid slice number."
                }
            )

        selected_slice = max(
            0,
            min(
                selected_slice,
                int(
                    case["num_slices"]
                ) - 1
            )
        )

        prediction = np.asarray(
            case["prediction"]
        )

        tumor_voxels = int(
            np.sum(
                prediction > 0
            )
        )

        result = (
            "TUMOR DETECTED"
            if tumor_voxels > 0
            else "NO TUMOR DETECTED"
        )

        report_path = make_pdf_report(
            case_id=case_id,
            case=case,
            result=result,
            selected_slice=selected_slice
        )

        return FileResponse(
            path=str(
                report_path
            ),
            media_type="application/pdf",
            filename=(
                f"{case_id}-brain-tumor-report.pdf"
            ),
            headers={
                "Cache-Control":
                    "no-store"
            }
        )

    except Exception as e:

        import traceback

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


# ============================================================
# DOWNLOAD SEGMENTATION
# ============================================================

@app.get("/case/{case_id}/download")
async def download_segmentation(
    case_id: str
):

    try:

        if case_id not in CASES:

            return JSONResponse(
                status_code=404,
                content={
                    "error":
                        "Case not found."
                }
            )

        case = CASES[
            case_id
        ]

        # ----------------------------------------------------
        # The existing prediction pipeline normally saves
        # the prediction inside user_predictions.
        # ----------------------------------------------------

        prediction_file = (
            find_prediction_file(
                case_id
            )
        )

        if prediction_file is None:

            return JSONResponse(
                status_code=404,
                content={
                    "error":
                        "Prediction NIfTI file was not found."
                }
            )

        return FileResponse(
            path=str(
                prediction_file
            ),
            media_type="application/gzip",
            filename=(
                f"{case_id}-segmentation.nii.gz"
            )
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )