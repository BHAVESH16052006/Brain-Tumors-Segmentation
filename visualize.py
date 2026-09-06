import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ============================================================
# PATHS
# ============================================================

PREDICTION_DIR = r"C:\Users\chand\BrainTumorProject\Brain-Tumors-Segmentation\test_predictions"

DATASET_DIR = r"C:\Users\chand\BrainTumorProject\dataset\brats2023\test"

OUTPUT_DIR = r"C:\Users\chand\BrainTumorProject\Brain-Tumors-Segmentation\qualitative_results"


# ============================================================
# SETTINGS
# ============================================================

NUM_PATIENTS = 10

# None = automatically select the axial slice
# containing the largest tumor area in the ground truth.
SLICE_INDEX = None


# ============================================================
# DISPLAY SETTINGS
# ============================================================

# Background / panel colors
FIGURE_BACKGROUND = "white"
HEADER_COLOR = "#0B102B"
TEXT_COLOR = "#10162F"

# Tumor colors
# Label 0 = Background
# Label 1 = NCR/NET
# Label 2 = Edema
# Label 3 = Enhancing Tumor

NCR_COLOR = "#F05A5A"       # Coral / red
EDEMA_COLOR = "#7ED26F"     # Green
ET_COLOR = "#6666C7"        # Blue / purple


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD NIFTI
# ============================================================

def load_nii(path):

    image = sitk.ReadImage(path)

    array = sitk.GetArrayFromImage(image)

    return array


# ============================================================
# NORMALIZE MRI
# ============================================================

def normalize_image(image):

    image = image.astype(np.float32)

    nonzero = image[image > 0]

    if len(nonzero) == 0:
        return image

    min_value = np.percentile(nonzero, 1)

    max_value = np.percentile(nonzero, 99)

    image = np.clip(
        image,
        min_value,
        max_value
    )

    image = (
        image - min_value
    ) / (
        max_value - min_value + 1e-8
    )

    return image


# ============================================================
# FIND BEST TUMOR SLICE
# ============================================================

def get_best_slice(label):

    # BraTS labels:
    #
    # 0 = Background
    # 1 = NCR/NET
    # 2 = Edema
    # 3 = Enhancing Tumor

    tumor = label > 0

    tumor_pixels = tumor.sum(axis=(1, 2))

    if tumor_pixels.max() == 0:

        return label.shape[0] // 2

    return int(np.argmax(tumor_pixels))


# ============================================================
# CREATE TUMOR COLORMAP
# ============================================================

def create_tumor_cmap():

    # Index 0 = transparent/background
    # Index 1 = NCR/NET
    # Index 2 = Edema
    # Index 3 = Enhancing Tumor

    cmap = ListedColormap([
        (0, 0, 0, 0),       # Background transparent
        NCR_COLOR,
        EDEMA_COLOR,
        ET_COLOR
    ])

    return cmap


TUMOR_CMAP = create_tumor_cmap()


# ============================================================
# DRAW MRI ONLY
# ============================================================

def draw_mri(ax, image):

    ax.imshow(
        image,
        cmap="gray",
        interpolation="nearest"
    )

    ax.axis("off")


# ============================================================
# DRAW MRI + SEGMENTATION OVERLAY
# ============================================================

def draw_segmentation(ax, image, label):

    # MRI background
    ax.imshow(
        image,
        cmap="gray",
        interpolation="nearest"
    )

    # Create transparent background
    masked_label = np.ma.masked_where(
        label == 0,
        label
    )

    # Colored tumor regions
    ax.imshow(
        masked_label,
        cmap=TUMOR_CMAP,
        vmin=0,
        vmax=3,
        alpha=0.55,
        interpolation="nearest"
    )

    ax.axis("off")


# ============================================================
# ADD PANEL HEADER
# ============================================================

def add_panel_header(fig, ax, text):

    position = ax.get_position()

    x = position.x0

    width = position.width

    y = position.y1 + 0.012

    height = 0.052

    header_ax = fig.add_axes([
        x,
        y,
        width,
        height
    ])

    header_ax.set_facecolor(HEADER_COLOR)

    # Rounded rectangle effect
    for spine in header_ax.spines.values():

        spine.set_visible(False)

    header_ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        color="white",
        fontsize=17,
        fontweight="bold"
    )

    header_ax.set_xticks([])

    header_ax.set_yticks([])


# ============================================================
# ADD PANEL BORDER
# ============================================================

def add_panel_border(ax):

    for spine in ax.spines.values():

        spine.set_visible(True)

        spine.set_linewidth(1.2)

        spine.set_edgecolor("#B8BDC9")


# ============================================================
# ADD LEGEND
# ============================================================

def add_legend(fig):

    legend_handles = [

        Patch(
            facecolor=EDEMA_COLOR,
            edgecolor="black",
            label="2 = Edema"
        ),

        Patch(
            facecolor=NCR_COLOR,
            edgecolor="black",
            label="1 = NCR/NET"
        ),

        Patch(
            facecolor=ET_COLOR,
            edgecolor="black",
            label="3 = Enhancing Tumor"
        ),

        Patch(
            facecolor="white",
            edgecolor="black",
            label="0 = Background"
        )
    ]

    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=4,
        frameon=True,
        fancybox=True,
        fontsize=13,
        handlelength=2.2,
        handleheight=1.3,
        columnspacing=2.5,
        handletextpad=0.7
    )

    legend.get_frame().set_facecolor("white")

    legend.get_frame().set_edgecolor("#C5CAD5")

    legend.get_frame().set_linewidth(1.0)

    # Explanation below legend
    fig.text(
        0.5,
        0.025,
        "Label Map: 0 = Background, 1 = NCR/NET, "
        "2 = Edema, 3 = Enhancing Tumor",
        ha="center",
        va="center",
        fontsize=12,
        color=TEXT_COLOR,
        style="italic"
    )


# ============================================================
# GET PREDICTIONS
# ============================================================

prediction_files = sorted(
    glob.glob(
        os.path.join(
            PREDICTION_DIR,
            "*-t1n.nii.gz"
        )
    )
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("QUALITATIVE SEGMENTATION VISUALIZATION")
print("=" * 70)

print("Prediction folder:")
print(PREDICTION_DIR)

print()

print("Dataset folder:")
print(DATASET_DIR)

print()

print("Predictions found:", len(prediction_files))

print("=" * 70)


if len(prediction_files) == 0:

    print()
    print("ERROR: No prediction files found.")

    raise SystemExit


# Limit number of patients
prediction_files = prediction_files[:NUM_PATIENTS]


# ============================================================
# PROCESS PATIENTS
# ============================================================

for patient_number, prediction_path in enumerate(
    prediction_files,
    start=1
):

    filename = os.path.basename(prediction_path)

    patient_id = filename.replace(
        "-t1n.nii.gz",
        ""
    )

    print()

    print(
        f"[{patient_number}/{len(prediction_files)}] "
        f"Processing {patient_id}"
    )


    # --------------------------------------------------------
    # Patient folder
    # --------------------------------------------------------

    patient_dir = os.path.join(
        DATASET_DIR,
        patient_id
    )


    # --------------------------------------------------------
    # T1n MRI
    # --------------------------------------------------------

    t1_path = os.path.join(
        patient_dir,
        f"{patient_id}-t1n.nii.gz"
    )


    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    gt_path = os.path.join(
        patient_dir,
        f"{patient_id}-seg.nii.gz"
    )


    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(t1_path):

        print("  ERROR: T1n MRI not found:")

        print(" ", t1_path)

        continue


    if not os.path.exists(gt_path):

        print("  ERROR: Ground truth not found:")

        print(" ", gt_path)

        continue


    # --------------------------------------------------------
    # Load NIFTI files
    # --------------------------------------------------------

    try:

        mri = load_nii(t1_path)

        ground_truth = load_nii(gt_path)

        prediction = load_nii(prediction_path)

    except Exception as e:

        print("  ERROR loading NIFTI:")

        print(" ", e)

        continue


    print(
        "  MRI shape        :",
        mri.shape
    )

    print(
        "  Ground truth     :",
        ground_truth.shape
    )

    print(
        "  Prediction       :",
        prediction.shape
    )


    # --------------------------------------------------------
    # Shape check
    # --------------------------------------------------------

    if (
        mri.shape != ground_truth.shape
        or
        mri.shape != prediction.shape
    ):

        print()

        print("  WARNING: Shape mismatch.")

        print("  Skipping patient.")

        continue


    # --------------------------------------------------------
    # Normalize MRI
    # --------------------------------------------------------

    mri = normalize_image(mri)


    # --------------------------------------------------------
    # Make sure labels are integer
    # --------------------------------------------------------

    ground_truth = ground_truth.astype(np.uint8)

    prediction = prediction.astype(np.uint8)


    # --------------------------------------------------------
    # Select identical axial slice
    # --------------------------------------------------------

    if SLICE_INDEX is None:

        slice_index = get_best_slice(
            ground_truth
        )

    else:

        slice_index = SLICE_INDEX


    # Safety check
    slice_index = max(
        0,
        min(
            slice_index,
            mri.shape[0] - 1
        )
    )


    print(
        "  Selected axial slice:",
        slice_index
    )


    # --------------------------------------------------------
    # Extract EXACT SAME slice
    # --------------------------------------------------------

    mri_slice = mri[slice_index]

    gt_slice = ground_truth[slice_index]

    pred_slice = prediction[slice_index]


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 8.5),
        facecolor=FIGURE_BACKGROUND
    )


    # --------------------------------------------------------
    # Reduce spacing
    # --------------------------------------------------------

    plt.subplots_adjust(
        left=0.025,
        right=0.975,
        top=0.78,
        bottom=0.19,
        wspace=0.035
    )


    # ========================================================
    # MRI
    # ========================================================

    draw_mri(
        axes[0],
        mri_slice
    )

    add_panel_border(
        axes[0]
    )


    # ========================================================
    # GROUND TRUTH
    # ========================================================

    draw_segmentation(
        axes[1],
        mri_slice,
        gt_slice
    )

    add_panel_border(
        axes[1]
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    draw_segmentation(
        axes[2],
        mri_slice,
        pred_slice
    )

    add_panel_border(
        axes[2]
    )


    # ========================================================
    # PANEL HEADERS
    # ========================================================

    add_panel_header(
        fig,
        axes[0],
        "MRI (T1n)"
    )

    add_panel_header(
        fig,
        axes[1],
        "Ground Truth"
    )

    add_panel_header(
        fig,
        axes[2],
        "Predicted Segmentation"
    )


    # ========================================================
    # MAIN TITLE
    # ========================================================

    fig.suptitle(
        f"{patient_id}  |  Axial Slice {slice_index}",
        fontsize=23,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.955
    )


    # ========================================================
    # LEGEND
    # ========================================================

    add_legend(fig)


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{patient_id}_qualitative.png"
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=FIGURE_BACKGROUND
    )


    plt.close(fig)


    print(
        "  Saved:",
        output_path
    )


# ============================================================
# FINISHED
# ============================================================

print()

print("=" * 70)

print("QUALITATIVE VISUALIZATION COMPLETED")

print("=" * 70)

print()

print("Output folder:")

print(OUTPUT_DIR)

print()

print(
    "Generated images can now be used "
    "for your thesis/report."
)

print()

print("=" * 70)