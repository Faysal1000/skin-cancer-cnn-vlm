"""
SkinCancerVisionXAI -- Gradio Application
==========================================
A research-grade skin lesion classification tool built on a CBAM-Xception
convolutional neural network.  The interface provides:

    - Six-stage image preprocessing (hair removal, illumination correction,
      CLAHE, resize, and Xception normalisation).
    - Classification into seven dermatoscopic categories from the HAM10000
      taxonomy (akiec, bcc, bkl, df, mel, nv, vasc).
    - Grad-CAM++ attention heatmap overlay showing which spatial regions
      influenced the classifier's decision.
    - Per-class confidence bar chart.
    - Optional VLM-based clinical explanation via MedGemma-4B + LoRA adapter.

This module is designed for deployment on Hugging Face Spaces (ZeroGPU compatible)
and also runs locally on CPU-only machines.

Authors : Faysal Ahmmed, Ajmy Alaly, Samanta Mehnaj, Asef Rahman, F.M. Mridha
License : Research use only -- not for clinical diagnosis.
"""

# =============================================================================
# Standard library imports
# =============================================================================
import os
import sys
import io
import glob
import gc
import signal
import traceback
from pathlib import Path


# =============================================================================
# Global safety net -- catch any uncaught exception or OOM at process level
# =============================================================================

def _global_exception_handler(exc_type, exc_value, exc_tb):
    """Last-resort handler for uncaught exceptions."""
    print(f"FATAL  Uncaught {exc_type.__name__}: {exc_value}")
    traceback.print_exception(exc_type, exc_value, exc_tb)
    if exc_type in (MemoryError,) or "out of memory" in str(exc_value).lower():
        print("FATAL  Memory error detected -- forcing restart ...")
        os._exit(1)

sys.excepthook = _global_exception_handler


def _sigterm_handler(signum, frame):
    """Handle SIGTERM gracefully (container shutdown)."""
    print(f"SIGNAL  Received signal {signum} -- shutting down cleanly ...")
    gc.collect()
    sys.exit(0)

signal.signal(signal.SIGTERM, _sigterm_handler)

# -----------------------------------------------------------------------------
# Python 3.13 removed the audioop module from the standard library.
# The audioop-lts package provides a drop-in replacement.  We inject it
# into sys.modules so that downstream libraries (e.g. pydub) that import
# audioop continue to work without modification.
# -----------------------------------------------------------------------------
try:
    import audioop
except ImportError:
    try:
        import audioop_lts as audioop
        sys.modules["audioop"] = audioop
    except ImportError:
        pass

# =============================================================================
# Third-party imports (CPU-safe, no GPU allocation required)
# =============================================================================
import numpy as np
import cv2
from PIL import Image

# Force the non-interactive Agg backend so matplotlib never tries to
# open a GUI window (important for headless servers and HF Spaces).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr

# =============================================================================
# ZeroGPU compatibility layer
# =============================================================================
# Hugging Face ZeroGPU Spaces expose a `spaces` module with a @spaces.GPU
# decorator that allocates a GPU on demand.  On CPU-only deployments or
# local machines, the module is absent.  We create a lightweight no-op
# decorator so the rest of the code can use @spaces.GPU unconditionally.

class _NoopSpaces:
    """Fallback when the real ZeroGPU runtime is not available."""

    @staticmethod
    def GPU(fn=None, **kw):
        if fn is not None:
            return fn
        def wrapper(f):
            return f
        return wrapper

try:
    import spaces
    _space_id = os.getenv("SPACE_ID", "")
    _zerogpu  = os.getenv("ZEROGPU", "")
    if _space_id and not _zerogpu:
        # Running on an HF Space that does NOT have ZeroGPU enabled.
        # Using the real @spaces.GPU decorator would cause a runtime error,
        # so we replace it with the no-op version.
        print("INFO  Running on HF Space without ZeroGPU -- @spaces.GPU disabled.")
        spaces = _NoopSpaces()
        HAS_ZEROGPU = False
    else:
        HAS_ZEROGPU = True
except ImportError:
    spaces = _NoopSpaces()
    HAS_ZEROGPU = False

# =============================================================================
# TensorFlow and Keras imports
# =============================================================================
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    Layer,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Reshape,
    Multiply,
    Add,
    Activation,
    Concatenate,
)
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# =============================================================================
# Application constants
# =============================================================================

# Input dimensions expected by the CBAM-Xception model
IMG_H, IMG_W = 299, 299

# Filesystem paths
MODEL_PATH  = "cbam-xception.keras"
LORA_DIR    = "./medgemma_lora_adapter"
HF_MODEL_ID = "google/medgemma-4b-it"

# Internal short-code labels (match the model output order)
LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Full clinical names displayed in the main prediction panel
FULL_NAMES = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}

# Abbreviated labels used on the confidence bar chart (saves horizontal space)
CHART_LABELS = {
    "akiec": "Actinic Keratoses",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}

# Classes that the dermatology literature considers malignant or pre-malignant
MALIGNANT = {"akiec", "bcc", "mel"}


# =============================================================================
# Dynamic example image discovery
# =============================================================================

def _discover_examples():
    """
    Scan the images/ directory for sample dermoscopy photographs.

    Returns a list-of-lists in the format required by gr.Examples.
    Supports .jpg, .jpeg, and .png extensions (case-insensitive).
    Falls back to a single hardcoded path if the directory is empty
    or missing so that the Examples widget never breaks.
    """
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    found = []
    for ext in extensions:
        found.extend(glob.glob(os.path.join("images", ext)))

    # Deduplicate (case-insensitive filesystems can return duplicates) and sort
    seen, unique = set(), []
    for path in sorted(found):
        normalised = os.path.normcase(path)
        if normalised not in seen:
            seen.add(normalised)
            unique.append(path)

    return [[p] for p in unique] if unique else [["images/ISIC_0025314.jpg"]]


EXAMPLE_PATHS = _discover_examples()


# =============================================================================
# Custom Keras layers and metrics
# =============================================================================
# Every custom object used when the model was trained must be registered here
# so that tf.keras.models.load_model() can reconstruct the graph from the
# saved .keras file.  Registration uses the @register_keras_serializable
# decorator, which adds the class/function to a global lookup table keyed
# by (package, name).

K = tf.keras.backend


@tf.keras.utils.register_keras_serializable(package="Custom", name="focal_ce_loss")
def focal_ce_loss(gamma=2.0, alpha=0.25, weight=0.3, num_classes=7):
    """
    Composite loss that blends focal loss with standard cross-entropy.

    Focal loss down-weights easy (well-classified) examples so that the
    network focuses its capacity on hard, misclassified samples.  The
    blending weight controls the contribution of each term.

    Parameters
    ----------
    gamma : float
        Focusing parameter.  Higher values increase the penalty on
        hard examples.
    alpha : float
        Balancing factor for the focal term.
    weight : float
        Blend ratio -- weight * focal + (1 - weight) * CE.
    num_classes : int
        Number of output classes (unused in the formula but kept for
        serialisation parity with the training code).
    """
    def _focal_ce(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        ce     = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        pt     = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal  = -alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt + K.epsilon())
        return tf.reduce_mean(weight * focal + (1.0 - weight) * ce)
    _focal_ce.__name__ = "focal_ce_loss"
    return _focal_ce


@tf.keras.utils.register_keras_serializable(package="Custom", name="F1Score")
class F1Score(tf.keras.metrics.Metric):
    """
    Harmonic mean of precision and recall, tracked as a Keras metric.

    This wraps the built-in Precision and Recall metrics and combines
    them on the fly so that F1 can be monitored during training without
    requiring a custom training loop.
    """

    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall    = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * p * r / (p + r + K.epsilon())

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

    def get_config(self):
        return super().get_config()


# -----------------------------------------------------------------------------
# CBAM Channel Attention
# -----------------------------------------------------------------------------
# Convolutional Block Attention Module (CBAM) applies channel attention
# followed by spatial attention.  The channel attention branch uses a
# shared two-layer MLP applied to both the average-pooled and max-pooled
# feature maps, then sums the results before a sigmoid gate.

@tf.keras.utils.register_keras_serializable(package="Custom", name="CBAMChannelAttention")
class CBAMChannelAttention(Layer):
    """
    CBAM channel attention module.

    Given a feature map of shape (B, H, W, C), this layer produces a
    (B, 1, 1, C) attention vector that re-weights each channel according
    to its global importance.
    """

    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels   = input_shape[-1]
        bottleneck = max(1, channels // self.reduction)
        self.fc1 = Dense(bottleneck, activation="relu",
                         kernel_initializer="he_normal", use_bias=True)
        self.fc2 = Dense(channels, kernel_initializer="he_normal", use_bias=True)
        super().build(input_shape)

    def call(self, x):
        avg = self.fc2(self.fc1(GlobalAveragePooling2D()(x)))
        mx  = self.fc2(self.fc1(GlobalMaxPooling2D()(x)))
        att = Activation("sigmoid")(Add()([avg, mx]))
        att = Reshape((1, 1, -1))(att)
        return Multiply()([x, att])

    def get_config(self):
        cfg = super().get_config()
        cfg["reduction"] = self.reduction
        return cfg


# Backwards-compatible alias so older saved models that reference the
# shorter class name can still be loaded without errors.
@tf.keras.utils.register_keras_serializable(package="Custom", name="ChannelAttention")
class ChannelAttention(CBAMChannelAttention):
    pass


# -----------------------------------------------------------------------------
# CBAM Spatial Attention
# -----------------------------------------------------------------------------
# The spatial branch concatenates the channel-wise average and max along
# the last axis, then applies a single convolution to produce a 2-D
# attention map that highlights informative spatial regions.

@tf.keras.utils.register_keras_serializable(package="Custom", name="CBAMSpatialAttention")
class CBAMSpatialAttention(Layer):
    """
    CBAM spatial attention module.

    Produces a single-channel spatial attention map of shape (B, H, W, 1)
    that scales each spatial position of the input feature map.
    """

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = Conv2D(1, kernel_size, padding="same", activation="sigmoid")

    def call(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx  = tf.reduce_max(x,  axis=-1, keepdims=True)
        return Multiply()([x, self.conv(Concatenate(axis=-1)([avg, mx]))])

    def get_config(self):
        cfg = super().get_config()
        cfg["kernel_size"] = self.kernel_size
        return cfg


# Backwards-compatible alias for older checkpoints.
@tf.keras.utils.register_keras_serializable(package="Custom", name="SpatialAttention")
class SpatialAttention(CBAMSpatialAttention):
    pass



# =============================================================================
# Offline preprocessing pipeline
# =============================================================================
# These three stages replicate the exact preprocessing applied during
# training (Notebook-01).  Applying them at inference time ensures that
# the model receives input with the same statistical properties it
# learned from.

def _dull_razor(img_bgr):
    """
    Remove dark hair artifacts from a dermoscopy image.

    Uses morphological black-hat filtering to isolate thin dark structures
    (hair), thresholds the result into a binary mask, dilates slightly to
    cover the full width of each hair strand, then inpaints the masked
    regions with TELEA interpolation to restore the underlying skin texture.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR colour order (OpenCV convention).

    Returns
    -------
    np.ndarray
        Hair-removed image in BGR colour order.
    """
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    mask = cv2.adaptiveThreshold(
        blackhat, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, -5,
    )
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)


def _illumination_correction(img_bgr, kernel_size=55, alpha=1.5, beta=-0.5):
    """
    Correct uneven illumination across the image.

    Converts to LAB colour space, estimates the background illumination
    of the L (lightness) channel via Gaussian blur, then subtracts it
    using a weighted blend.  This flattens brightness gradients caused
    by uneven lighting while preserving local contrast.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR colour order.
    kernel_size : int
        Size of the Gaussian kernel used to estimate the illumination
        background.  Must be odd.
    alpha, beta : float
        Blending weights for cv2.addWeighted(L, alpha, background, beta, 0).

    Returns
    -------
    np.ndarray
        Illumination-corrected image in BGR colour order.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    background  = cv2.GaussianBlur(l, (kernel_size, kernel_size), 0)
    l_corrected = cv2.addWeighted(l, alpha, background, beta, 0)
    l_corrected = np.clip(l_corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l_corrected, a, b)), cv2.COLOR_LAB2BGR)


def _apply_clahe(img_bgr, clip_limit=0.02, tile_grid=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalisation (CLAHE).

    Operates on the L channel of the LAB colour space so that only
    lightness is enhanced -- hue and saturation remain unchanged.
    The clip limit prevents over-amplification of noise in flat regions.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR colour order.
    clip_limit : float
        Threshold for contrast limiting.
    tile_grid : tuple of int
        Grid size for the CLAHE algorithm.

    Returns
    -------
    np.ndarray
        Contrast-enhanced image in BGR colour order.
    """
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq    = clahe.apply(l)
    lab_eq  = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# =============================================================================
# CNN preprocessing (matches Notebook-03 training pipeline)
# =============================================================================

# Xception's preprocess_input rescales pixel values from [0, 255] to [-1, 1].
_XCEPTION_PREPROCESS = tf.keras.applications.xception.preprocess_input


def _to_display(arr_minus1_1):
    """Convert a float32 array in [-1, 1] back to uint8 in [0, 255] for display."""
    return np.clip((arr_minus1_1 + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)


def preprocess_image(pil_image: Image.Image):
    """
    Apply the complete six-stage preprocessing pipeline to a PIL image.

    Stages
    ------
    1. DullRazor hair removal
    2. Illumination correction (Gaussian background subtraction)
    3. CLAHE contrast enhancement
    4. Resize to (IMG_H, IMG_W)
    5. Xception normalisation (scales pixels to [-1, 1])
    6. Batch dimension expansion

    Returns
    -------
    input_tensor : np.ndarray
        Shape (1, IMG_H, IMG_W, 3), float32, ready for model.predict().
    steps : dict
        Mapping from stage name to PIL image, used to populate the
        preprocessing pipeline gallery in the UI.
    """
    orig_rgb = np.array(pil_image.convert("RGB"))
    bgr      = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    # Stage 1 -- Hair removal
    hair_bgr = _dull_razor(bgr)
    hair_rgb = cv2.cvtColor(hair_bgr, cv2.COLOR_BGR2RGB)

    # Stage 2 -- Illumination correction
    illum_bgr = _illumination_correction(hair_bgr)
    illum_rgb = cv2.cvtColor(illum_bgr, cv2.COLOR_BGR2RGB)

    # Stage 3 -- CLAHE contrast enhancement
    clahe_bgr = _apply_clahe(illum_bgr)
    clahe_rgb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)

    # Stage 4 -- Resize to model input dimensions
    resized = cv2.resize(clahe_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

    # Stage 5 -- Xception normalisation (pixels to [-1, 1])
    preproc_f    = _XCEPTION_PREPROCESS(resized.astype(np.float32).copy())
    input_tensor = np.expand_dims(preproc_f, axis=0)

    # Collect intermediate images for the preprocessing gallery
    steps = {
        "original":        pil_image,
        "hair_removed":    Image.fromarray(hair_rgb),
        "illum_corrected": Image.fromarray(illum_rgb),
        "clahe_applied":   Image.fromarray(clahe_rgb),
        "resized":         Image.fromarray(resized),
        "normalized":      Image.fromarray(_to_display(preproc_f)),
    }
    return input_tensor, steps


# =============================================================================
# Model cache (module-level singletons, persist across requests)
# =============================================================================

_cnn_model  = None   # Cached CBAM-Xception Keras model
_vlm_bundle = None   # Cached VLM processor + model + device info


def _load_cnn():
    """
    Load and cache the CBAM-Xception classifier from disk.

    The model is loaded once on the first request and reused for all
    subsequent inference calls within the same worker process.
    """
    global _cnn_model
    if _cnn_model is None:
        _cnn_model = load_model(MODEL_PATH)
    return _cnn_model


def _load_vlm():
    """
    Load and cache the MedGemma-4B model with the fine-tuned LoRA adapter.

    Uses 4-bit NF4 quantisation on CUDA-capable GPUs to reduce memory
    usage.  Falls back to full float32 on CPU (functional but slow).

    Returns
    -------
    dict with keys 'model', 'processor', 'device'.
    """
    global _vlm_bundle
    if _vlm_bundle is not None:
        return _vlm_bundle

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from peft import PeftModel

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("WARNING  No HF_TOKEN found. Gated model downloads may fail.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        capability = torch.cuda.get_device_capability(0)[0]
        dtype      = torch.bfloat16 if capability >= 8 else torch.float32
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        # Use bfloat16 to halve memory (~4.3 GB instead of ~8.6 GB).
        # bfloat16 (not float16!) is essential on CPU -- float16 has only
        # 5 exponent bits which causes NaN/underflow in attention layers,
        # producing empty outputs.  bfloat16 has 8 exponent bits (same
        # as float32) so it's numerically stable.
        dtype      = torch.bfloat16
        bnb_config = None
        print("INFO  GPU not detected. VLM will run on CPU in bfloat16 (memory-efficient).")

    if not os.path.isdir(LORA_DIR):
        raise FileNotFoundError(
            f"LoRA adapter directory not found: {LORA_DIR}\n"
            "Please download the adapter and place it in the project root."
        )

    processor = AutoProcessor.from_pretrained(
        LORA_DIR, trust_remote_code=True, token=hf_token
    )
    processor.tokenizer.padding_side = "right"

    load_kwargs = dict(
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,       # avoid 2x memory spike during load
    )
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"]          = "auto"
    else:
        load_kwargs["dtype"] = dtype

    base  = AutoModelForImageTextToText.from_pretrained(HF_MODEL_ID, **load_kwargs)
    model = PeftModel.from_pretrained(
        base, LORA_DIR,
        **({} if bnb_config is None else {"device_map": "auto"}),
    )
    model.eval()

    _vlm_bundle = {"model": model, "processor": processor, "device": device}
    return _vlm_bundle


# =============================================================================
# Grad-CAM++ target layer selection
# =============================================================================

def _find_gradcam_layer(mdl):
    """
    Locate the best convolutional layer for Grad-CAM++ visualisation.

    Searches the model in reverse layer order for the last CBAM-related
    layer with a 4-D output tensor (batch, height, width, channels).
    Falls back to the last Conv2D layer, and ultimately to the Xception
    default 'block14_sepconv2_act' if nothing else is found.

    Parameters
    ----------
    mdl : tf.keras.Model
        The loaded CBAM-Xception model.

    Returns
    -------
    str
        Name of the target layer for Grad-CAM++.
    """
    cbam_keywords = ["cbam", "channel_attention", "spatial_attention", "multiply"]
    for layer in reversed(mdl.layers):
        name = layer.name.lower()
        if any(kw in name for kw in cbam_keywords):
            if len(layer.output.shape) == 4:
                return layer.name

    # Fallback: last Conv2D in the network
    last_conv = None
    for layer in mdl.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
    return last_conv or "block14_sepconv2_act"


# =============================================================================
# Probability bar chart
# =============================================================================

def _make_bar_chart(probs):
    """
    Render a horizontal bar chart showing per-class prediction confidence.

    Malignant / pre-malignant classes are coloured in warm red tones;
    benign classes use a cooler blue.  Percentage labels are drawn to
    the right of each bar for readability.

    Parameters
    ----------
    probs : list of float
        Seven confidence values in the same order as LABELS.

    Returns
    -------
    PIL.Image.Image
        The rendered chart as a PIL image.
    """
    chart_names = [CHART_LABELS[lbl] for lbl in LABELS]
    colors      = ["#e05555" if lbl in MALIGNANT else "#4a90d2" for lbl in LABELS]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    bars = ax.barh(chart_names, probs, color=colors, edgecolor="none", height=0.55)

    # Draw percentage labels beside each bar
    for bar, p in zip(bars, probs):
        x_pos = min(p + 0.015, 0.90)
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f"{p * 100:.1f}%",
            va="center", ha="left", fontsize=9, color="#1a1a2e", fontweight="medium",
        )

    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Confidence", fontsize=10, color="#444", labelpad=8)
    ax.set_title(
        "Prediction Confidence by Class",
        fontsize=11.5, fontweight="bold", pad=12, color="#1a1a2e",
    )
    ax.tick_params(axis="y", labelsize=9.5, colors="#333")
    ax.tick_params(axis="x", labelsize=8.5, colors="#555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.set_facecolor("#f7f9fc")
    fig.patch.set_facecolor("#f7f9fc")
    plt.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# =============================================================================
# Core inference function
# =============================================================================
# NOTE: TensorFlow CNN inference runs on CPU, so @spaces.GPU is intentionally
# NOT applied here.  This means the "Run Analysis" button works on every
# deployment type -- CPU-only HF Spaces, local machines, and ZeroGPU Spaces.

def run_analysis(image: Image.Image, progress=gr.Progress(track_tqdm=True)):
    """
    Execute the full analysis pipeline on a single dermoscopy image.

    Pipeline steps
    --------------
    1-3.  Offline preprocessing (hair removal, illumination, CLAHE)
    4-5.  CNN preprocessing (resize, Xception normalisation)
      6.  CBAM-Xception classification
      7.  Grad-CAM++ attention heatmap generation
      8.  Prediction text formatting
      9.  Confidence bar chart rendering

    Returns
    -------
    tuple
        Eleven values matching the order of btn_analyze.click outputs:
        six preprocessing step images, the heatmap overlay, prediction
        markdown text, the bar chart image, and two state values
        (pred_label, overlay_pil) consumed by the VLM step.
    """
    if image is None:
        return (
            (None,) * 6
            + (None, "Please upload or select an image before running analysis.", None, None, None)
        )

    try:
        # Steps 1-5: full preprocessing pipeline
        progress(0.05, desc="Applying preprocessing pipeline ...")
        input_tensor, steps = preprocess_image(image)

        # Step 6: load classifier and run forward pass
        progress(0.25, desc="Loading classifier ...")
        cnn = _load_cnn()
        progress(0.45, desc="Running classification ...")
        preds      = cnn.predict(input_tensor, verbose=0)[0]
        pred_idx   = int(np.argmax(preds))
        pred_label = LABELS[pred_idx]
        conf       = float(preds[pred_idx])

        # Step 7: generate Grad-CAM++ heatmap
        progress(0.62, desc="Generating attention heatmap ...")
        target_layer = _find_gradcam_layer(cnn)
        score        = CategoricalScore([pred_idx])
        gradcam_vis  = GradcamPlusPlus(cnn, model_modifier=ReplaceToLinear(), clone=True)
        cam          = gradcam_vis(score, input_tensor, penultimate_layer=target_layer)[0]
        cam          = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Blend the heatmap over the resized image for display
        resized_arr = np.array(steps["resized"]).astype(np.float32) / 255.0
        heatmap     = plt.cm.jet(cam)[..., :3]
        overlay_arr = np.clip(0.35 * heatmap + 0.65 * resized_arr, 0, 1)
        overlay_pil = Image.fromarray((overlay_arr * 255).astype(np.uint8))

        # Step 8: format the prediction result as Markdown
        is_malignant = pred_label in MALIGNANT
        risk_label   = "MALIGNANT / PRE-MALIGNANT" if is_malignant else "BENIGN"
        if is_malignant:
            risk_note = (
                "> **[!] Warning:** This class is considered malignant or pre-malignant. "
                "Please consult a qualified dermatologist as soon as possible.\n"
            )
        else:
            risk_note = (
                "> This class is generally considered benign. "
                "Always verify findings with a qualified dermatologist.\n"
            )

        pred_md = (
            f"### {FULL_NAMES[pred_label]}\n\n"
            #f"**Risk Category:** {risk_label}\n\n"
            f"**Confidence:** {conf * 100:.1f}%\n\n"
            + risk_note
        )

        # Step 9: render the confidence bar chart
        progress(0.88, desc="Building confidence chart ...")
        bar_chart = _make_bar_chart(preds.tolist())

        progress(1.0, desc="Analysis complete.")
        return (
            steps["original"],
            steps["hair_removed"],
            steps["illum_corrected"],
            steps["clahe_applied"],
            steps["resized"],
            steps["normalized"],
            overlay_pil,
            pred_md,
            bar_chart,
            pred_label,   # stored in gr.State for the VLM step
            overlay_pil,  # stored in gr.State for the VLM step
        )

    except Exception:
        err = traceback.format_exc()
        return (
            (None,) * 6
            + (None, f"**An error occurred during analysis:**\n\n```\n{err}\n```", None, None, None)
        )


# =============================================================================
# VLM clinical explanation
# =============================================================================

@spaces.GPU(duration=180)
def run_vlm(overlay_pil: Image.Image, pred_label: str, progress=gr.Progress(track_tqdm=True)):
    """
    Generate a short clinical explanation using MedGemma-4B + LoRA.

    This is a **generator function** -- it yields intermediate status
    messages to keep the WebSocket connection alive during the potentially
    long model-loading phase on CPU.  The final yield is the actual
    clinical explanation (or an error message).

    Parameters
    ----------
    overlay_pil : PIL.Image.Image
        The Grad-CAM++ heatmap overlay produced by run_analysis().
    pred_label : str
        The short-code label predicted by the CNN (e.g. "mel").

    Yields
    ------
    str
        Status updates during loading, then the final explanation.
    """
    if overlay_pil is None or pred_label is None:
        yield "Please run the main analysis first before requesting a VLM explanation."
        return

    try:
        # --- Phase 0: Free CNN memory before loading VLM ---
        # The CNN (~2-3 GB) + VLM (~8.6 GB) together exceed the 16 GB
        # memory limit.  We unload the CNN first; it will be re-loaded
        # automatically next time the user clicks "Run Analysis".
        global _cnn_model
        if _cnn_model is not None:
            print("VLM  Unloading CNN to free memory for VLM ...")
            _cnn_model = None
            import gc
            gc.collect()
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except Exception:
                pass

        # --- Phase 1: Load the VLM model (slow on CPU) ---
        print("VLM  Starting model load (MedGemma-4B) ...")
        yield "Loading VLM model (MedGemma-4B) -- please wait ..."
        progress(0.05, desc="Loading VLM model (MedGemma-4B) ...")

        import torch
        import threading
        import time

        # Run model loading in a background thread so we can yield
        # periodic heartbeat messages that keep the connection alive.
        load_result = {"bundle": None, "error": None}

        def _background_load():
            try:
                load_result["bundle"] = _load_vlm()
            except Exception as exc:
                load_result["error"] = exc

        loader = threading.Thread(target=_background_load, daemon=True)
        loader.start()

        # Yield a status update every 10 seconds while loading.
        # This sends data over the WebSocket and prevents the HF
        # reverse proxy from timing out the connection.
        elapsed = 0
        while loader.is_alive():
            loader.join(timeout=10)
            elapsed += 10
            if loader.is_alive():
                mins = elapsed // 60
                secs = elapsed % 60
                yield f"Still loading VLM model ... ({mins}m {secs}s elapsed)"

        # Check for errors during loading
        if load_result["error"] is not None:
            raise load_result["error"]

        bundle    = load_result["bundle"]
        processor = bundle["processor"]
        model     = bundle["model"]
        device    = bundle["device"]

        # --- Phase 2: Prepare the prompt ---
        print("VLM  Model loaded successfully. Preparing prompt ...")
        yield "Model loaded. Preparing prompt and tokenising ..."
        progress(0.45, desc="Preparing prompt and tokenising ...")

        img_in = overlay_pil.convert("RGB").resize((299, 299), Image.BILINEAR)

        prompt_text = (
            "You are an AI assistant specialised in model interpretability for dermatology. "
            "I am providing a Grad-CAM++ heatmap overlay from a CBAM-Xception CNN classifier.\n"
            f"Predicted class: {FULL_NAMES[pred_label]}\n\n"
            "Write a clear, concise 25-40 word explanation of which visual features the model "
            "focused on and why that supports the prediction. Output only the explanation, "
            "with no preamble."
        )
        chat = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }]

        formatted = processor.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=formatted, images=img_in, return_tensors="pt", padding=True
        )
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        if "pixel_values" in inputs:
            try:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
            except Exception:
                pass

        # --- Phase 3: Generate the explanation (in background thread) ---
        # On CPU, model.generate() can block for several minutes.
        # We run it in a thread and yield heartbeats to keep the
        # WebSocket alive, same pattern as Phase 1.
        print("VLM  Generating clinical explanation ...")
        yield "Generating clinical explanation ..."
        progress(0.70, desc="Generating clinical explanation ...")

        gen_result = {"output": None, "error": None}

        def _background_generate():
            try:
                with torch.inference_mode():
                    gen_result["output"] = model.generate(
                        **inputs,
                        max_new_tokens=80,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )
            except Exception as exc:
                gen_result["error"] = exc

        gen_thread = threading.Thread(target=_background_generate, daemon=True)
        gen_thread.start()

        gen_elapsed = 0
        while gen_thread.is_alive():
            gen_thread.join(timeout=10)
            gen_elapsed += 10
            if gen_thread.is_alive():
                mins = gen_elapsed // 60
                secs = gen_elapsed % 60
                yield f"Generating clinical explanation ... ({mins}m {secs}s elapsed)"

        if gen_result["error"] is not None:
            raise gen_result["error"]

        out_ids  = gen_result["output"]
        seqs     = out_ids.sequences if hasattr(out_ids, "sequences") else out_ids
        in_len   = inputs["input_ids"].shape[-1]
        response = processor.decode(seqs[0, in_len:], skip_special_tokens=True).strip()

        # --- Phase 4: Unload VLM from RAM after generation ---
        # The CNN + VLM together exceed 16 GB.  We free the VLM now
        # so the CNN can be re-loaded on the next "Run Analysis" click.
        print("VLM  Unloading VLM model to free memory ...")
        _vlm_bundle = None
        del model, processor, bundle, inputs, out_ids, seqs
        import gc
        gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
        print("VLM  VLM model unloaded. Memory freed.")

        yield response or "The model returned an empty response."

    except (MemoryError, RuntimeError) as mem_err:
        # Catch OOM errors (CPU MemoryError or CUDA RuntimeError)
        err_msg = str(mem_err).lower()
        is_oom = isinstance(mem_err, MemoryError) or "out of memory" in err_msg
        if is_oom:
            print("VLM  *** OUT OF MEMORY -- clearing all models and restarting ***")
            _vlm_bundle = None
            _cnn_model = None
            import gc
            gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass
            # Force-restart: HF Spaces will auto-restart the container.
            print("VLM  Forcing process exit for auto-restart ...")
            os._exit(1)
        else:
            _vlm_bundle = None
            import gc
            gc.collect()
            yield f"VLM generation failed:\n\n```\n{traceback.format_exc()}\n```"

    except Exception:
        # Any other error: clear VLM to free memory, show error
        print(f"VLM  Unexpected error: {traceback.format_exc()}")
        _vlm_bundle = None
        _cnn_model = None
        import gc
        gc.collect()
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        yield f"VLM generation failed:\n\n```\n{traceback.format_exc()}\n```"



# =============================================================================
# CSS -- Premium dark theme with clean visual hierarchy
# =============================================================================
# The stylesheet uses Inter for body text and Playfair Display for headings.
# The colour palette centres on deep navy backgrounds (#0d1520, #111c2b) with
# cool blue accents (#3a9ec0, #5c50d8) and warm amber for warnings (#d4a830).
# Every section has generous padding and rounded corners so the interface
# feels spacious and readable on both desktop and mobile.

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

/* =============================================================
   Dark mode (default) -- override BOTH our custom vars AND
   Gradio's internal vars so everything is consistent
   ============================================================= */
.gradio-container {
    /* --- Our custom variables --- */
    --bg-primary:         #0a1018;
    --bg-secondary:       #0d1520;
    --bg-tertiary:        #111c2b;
    --bg-card:            rgba(255,255,255,0.025);
    --bg-card-hover:      rgba(255,255,255,0.04);
    --border-subtle:      rgba(255,255,255,0.06);
    --border-medium:      rgba(255,255,255,0.10);
    --text-primary:       #e4eef8;
    --text-secondary:     #d0e2f0;
    --text-muted:         #6a8faa;
    --text-helper:        #5a7d96;
    --heading-color:      #e8f2fc;
    --section-label:      #a0bdd4;
    --accent-blue:        #3a9ec0;
    --accent-purple:      #5c50d8;
    --btn-gradient-start: #2d8aab;
    --btn-gradient-end:   #4c44c4;
    --btn-shadow:         rgba(44, 138, 171, 0.25);
    --btn-shadow-hover:   rgba(44, 138, 171, 0.35);
    --vlm-bg:             rgba(92, 80, 216, 0.10);
    --vlm-border:         rgba(92, 80, 216, 0.30);
    --vlm-text:           #a898f0;
    --vlm-input-bg:       rgba(255,255,255,0.03);
    --vlm-input-text:     #e4eef8;
    --blockquote-bg:      rgba(58, 158, 192, 0.05);
    --blockquote-text:    #8fb8d0;
    --blockquote-border:  #3a9ec0;
    --accordion-bg:       rgba(255,255,255,0.018);
    --accordion-label:    #7da8c4;
    --step-bg:            rgba(255,255,255,0.015);
    --step-border:        rgba(255,255,255,0.04);
    --workflow-bg:        rgba(58, 158, 192, 0.08);
    --workflow-border:    rgba(58, 158, 192, 0.18);
    --workflow-text:      #7db8d0;
    --workflow-strong:    #a0d0e8;
    --warn-bg:            rgba(180, 120, 10, 0.06);
    --warn-border:        rgba(180, 120, 10, 0.15);
    --warn-border-left:   #b87e0a;
    --warn-text:          #c09830;
    --warn-strong:        #d4a830;
    --warn-copy:          #8a7530;
    --img-border:         rgba(255,255,255,0.05);
    --pred-border-bottom: rgba(255,255,255,0.06);

    /* --- Gradio internal variables (dark) --- */
    --body-background-fill:        #0d1520 !important;
    --background-fill-primary:     #111c2b !important;
    --background-fill-secondary:   #0d1520 !important;
    --block-background-fill:       rgba(255,255,255,0.025) !important;
    --block-border-color:          rgba(255,255,255,0.06) !important;
    --block-label-background-fill: rgba(255,255,255,0.04) !important;
    --block-label-text-color:      #a0bdd4 !important;
    --block-title-text-color:      #d0e2f0 !important;
    --body-text-color:             #d0e2f0 !important;
    --body-text-color-subdued:     #6a8faa !important;
    --border-color-accent:         #3a9ec0 !important;
    --border-color-primary:        rgba(255,255,255,0.06) !important;
    --color-accent:                #3a9ec0 !important;
    --color-accent-soft:           rgba(58,158,192,0.15) !important;
    --input-background-fill:       rgba(255,255,255,0.03) !important;
    --input-border-color:          rgba(255,255,255,0.08) !important;
    --input-placeholder-color:     #5a7d96 !important;
    --link-text-color:             #3a9ec0 !important;
    --panel-background-fill:       rgba(255,255,255,0.018) !important;
    --table-border-color:          rgba(255,255,255,0.06) !important;
    --button-secondary-background-fill: rgba(255,255,255,0.04) !important;
    --button-secondary-border-color:    rgba(255,255,255,0.08) !important;
    --button-secondary-text-color:      #d0e2f0 !important;

    /* --- Background gradient --- */
    background: linear-gradient(160deg, #0a1018 0%, #0d1520 40%, #111c2b 100%) !important;
    min-height: 100vh;
}

/* =============================================================
   Light mode -- .light-mode class overrides everything
   ============================================================= */
.gradio-container.light-mode {
    /* --- Our custom variables (light) --- */
    --bg-primary:         #f4f6f9 !important;
    --bg-secondary:       #edf0f5 !important;
    --bg-tertiary:        #e8ecf2 !important;
    --bg-card:            rgba(255,255,255,0.85) !important;
    --bg-card-hover:      rgba(255,255,255,1.0) !important;
    --border-subtle:      rgba(0,0,0,0.07) !important;
    --border-medium:      rgba(0,0,0,0.12) !important;
    --text-primary:       #1a1a2e !important;
    --text-secondary:     #2d3748 !important;
    --text-muted:         #5a6a7a !important;
    --text-helper:        #6b7b8d !important;
    --heading-color:      #1a1a2e !important;
    --section-label:      #4a6580 !important;
    --accent-blue:        #2980b9 !important;
    --accent-purple:      #5c50d8 !important;
    --vlm-input-text:     #2d3748 !important;
    --vlm-input-bg:       rgba(0,0,0,0.02) !important;
    --blockquote-text:    #1a6fa0 !important;
    --blockquote-bg:      rgba(41,128,185,0.06) !important;
    --blockquote-border:  #2980b9 !important;
    --accordion-bg:       rgba(255,255,255,0.6) !important;
    --accordion-label:    #4a6580 !important;
    --step-bg:            rgba(255,255,255,0.5) !important;
    --step-border:        rgba(0,0,0,0.06) !important;
    --workflow-bg:        rgba(41,128,185,0.06) !important;
    --workflow-border:    rgba(41,128,185,0.15) !important;
    --workflow-text:      #2980b9 !important;
    --workflow-strong:    #1a6fa0 !important;
    --warn-bg:            rgba(200,140,20,0.06) !important;
    --warn-border:        rgba(200,140,20,0.15) !important;
    --warn-border-left:   #c88c14 !important;
    --warn-text:          #8a6200 !important;
    --warn-strong:        #6d4e00 !important;
    --warn-copy:          #8a6200 !important;
    --img-border:         rgba(0,0,0,0.08) !important;
    --pred-border-bottom: rgba(0,0,0,0.08) !important;

    /* --- Gradio internal variables (light) --- */
    --body-background-fill:        #edf0f5 !important;
    --background-fill-primary:     #ffffff !important;
    --background-fill-secondary:   #f4f6f9 !important;
    --block-background-fill:       #ffffff !important;
    --block-border-color:          rgba(0,0,0,0.08) !important;
    --block-label-background-fill: #f4f6f9 !important;
    --block-label-text-color:      #4a6580 !important;
    --block-title-text-color:      #2d3748 !important;
    --body-text-color:             #2d3748 !important;
    --body-text-color-subdued:     #5a6a7a !important;
    --border-color-accent:         #2980b9 !important;
    --border-color-primary:        rgba(0,0,0,0.08) !important;
    --color-accent:                #2980b9 !important;
    --color-accent-soft:           rgba(41,128,185,0.10) !important;
    --input-background-fill:       #ffffff !important;
    --input-border-color:          rgba(0,0,0,0.10) !important;
    --input-placeholder-color:     #6b7b8d !important;
    --link-text-color:             #2980b9 !important;
    --panel-background-fill:       #f9fafb !important;
    --table-border-color:          rgba(0,0,0,0.08) !important;
    --button-secondary-background-fill: #f4f6f9 !important;
    --button-secondary-border-color:    rgba(0,0,0,0.10) !important;
    --button-secondary-text-color:      #2d3748 !important;
    --checkbox-label-text-color:        #2d3748 !important;
    --accordion-text-color:             #2d3748 !important;
    --neutral-700:                      #4a6580 !important;
    --neutral-800:                      #2d3748 !important;
    --shadow-drop:                      0 1px 3px rgba(0,0,0,0.06) !important;

    /* --- Background gradient (light) --- */
    background: linear-gradient(160deg, #f4f6f9 0%, #edf0f5 40%, #e8ecf2 100%) !important;
}

/* --- Light mode: force ALL text inside Gradio to be dark --- */
.gradio-container.light-mode label,
.gradio-container.light-mode .gr-accordion > .label-wrap,
.gradio-container.light-mode .gr-accordion > .label-wrap > span,
.gradio-container.light-mode .gr-accordion summary,
.gradio-container.light-mode .gr-accordion summary span,
.gradio-container.light-mode .gr-block label span,
.gradio-container.light-mode .gr-input-label,
.gradio-container.light-mode .gr-check-radio label,
.gradio-container.light-mode span.svelte-1gfkn6j,
.gradio-container.light-mode .label-wrap > span {
    color: #2d3748 !important;
}

/* Accordion panel background in light mode */
.gradio-container.light-mode .gr-accordion,
.gradio-container.light-mode details {
    background: rgba(255,255,255,0.7) !important;
    border-color: rgba(0,0,0,0.08) !important;
}

/* VLM and other textareas in light mode */
.gradio-container.light-mode textarea,
.gradio-container.light-mode input[type="text"] {
    color: #2d3748 !important;
    background: #ffffff !important;
}

/* Helper text in light mode */
.gradio-container.light-mode .helper-text,
.gradio-container.light-mode p.helper-text {
    color: #5a6a7a !important;
}

/* Prediction markdown in light mode */
.gradio-container.light-mode #pred-md,
.gradio-container.light-mode #pred-md * {
    color: #2d3748 !important;
}
.gradio-container.light-mode #pred-md h3 {
    color: #1a1a2e !important;
}
.gradio-container.light-mode #pred-md strong {
    color: #1a1a2e !important;
}
.gradio-container.light-mode #pred-md blockquote {
    color: #1a6fa0 !important;
    background: rgba(41,128,185,0.06) !important;
    border-left-color: #2980b9 !important;
}

/* Section titles in light mode */
.gradio-container.light-mode .section-title,
.gradio-container.light-mode .section-title * {
    color: #4a6580 !important;
}

/* Disclaimer in light mode */
.gradio-container.light-mode #disclaimer-banner {
    color: #8a6200 !important;
    background: rgba(200,140,20,0.06) !important;
}
.gradio-container.light-mode #disclaimer-banner strong {
    color: #6d4e00 !important;
}

/* VLM button in light mode */
.gradio-container.light-mode #btn-vlm button {
    background: rgba(92,80,216,0.06) !important;
    border-color: rgba(92,80,216,0.20) !important;
    color: #5c50d8 !important;
}

/* Workflow steps in light mode */
.gradio-container.light-mode .workflow-step {
    background: rgba(41,128,185,0.06) !important;
    border-color: rgba(41,128,185,0.15) !important;
    color: #2980b9 !important;
}
.gradio-container.light-mode .workflow-step strong {
    color: #1a6fa0 !important;
}

/* ===== Global reset and base typography ===== */
body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ===== Theme toggle button ===== */
#theme-toggle {
    position: fixed !important;
    top: 60px;
    right: 16px;
    z-index: 9999;
    max-width: 120px !important;
    min-width: 90px !important;
}
#theme-toggle button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #8faabb !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 0.9rem !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px !important;
    height: auto !important;
    min-height: unset !important;
}
#theme-toggle button:hover {
    background: rgba(255,255,255,0.14) !important;
    color: #c0d8e8 !important;
}
.light-mode #theme-toggle button,
.gradio-container.light-mode #theme-toggle button {
    background: rgba(0,0,0,0.05) !important;
    border-color: rgba(0,0,0,0.12) !important;
    color: #4a6580 !important;
}
.light-mode #theme-toggle button:hover,
.gradio-container.light-mode #theme-toggle button:hover {
    background: rgba(0,0,0,0.09) !important;
    color: #2d3748 !important;
}

/* ===== Application header ===== */
#app-header {
    position: relative;
    text-align: center;
    padding: 2.5rem 1.5rem 1.8rem;
    margin-bottom: 0.5rem;
}
#app-header h1 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.5rem;
    letter-spacing: 0.5px;
}
#app-header .subtitle {
    color: var(--text-muted);
    font-size: 0.92rem;
    font-weight: 400;
    letter-spacing: 0.2px;
    line-height: 1.6;
}
#app-header .divider {
    width: 80px;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
    border-radius: 2px;
    margin: 1rem auto 0;
}

/* ===== Section headers ===== */
.section-title {
    color: var(--section-label) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    margin: 1.5rem 0 0.8rem !important;
    padding-left: 2px !important;
}

/* ===== Input card styling ===== */
#upload-section {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    background: var(--bg-card) !important;
    padding: 0.3rem !important;
}

/* ===== Run Analysis button ===== */
#btn-analyze {
    background: linear-gradient(135deg, var(--btn-gradient-start) 0%, var(--btn-gradient-end) 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border-radius: 12px !important;
    height: 54px !important;
    letter-spacing: 0.4px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    margin-top: 0.8rem !important;
    box-shadow: 0 4px 20px var(--btn-shadow) !important;
}
#btn-analyze:hover {
    opacity: 0.9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px var(--btn-shadow-hover) !important;
}
#btn-analyze:active {
    transform: translateY(0) !important;
}

/* ===== Results panel ===== */
#results-card {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    background: var(--bg-card) !important;
    padding: 1.2rem 1.4rem !important;
    min-height: 120px;
}

/* ===== Prediction markdown ===== */
#pred-md, #pred-md * {
    color: var(--text-secondary) !important;
}
#pred-md h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 1.25rem !important;
    color: var(--heading-color) !important;
    margin-bottom: 0.7rem !important;
    line-height: 1.4 !important;
    border-bottom: 1px solid var(--pred-border-bottom);
    padding-bottom: 0.6rem;
}
#pred-md blockquote {
    border-left: 3px solid var(--blockquote-border);
    padding: 0.5rem 0 0.5rem 1rem;
    margin: 0.8rem 0 0;
    color: var(--blockquote-text) !important;
    font-size: 0.92rem !important;
    background: var(--blockquote-bg);
    border-radius: 0 6px 6px 0;
}
#pred-md strong {
    color: var(--heading-color) !important;
}

/* ===== Heatmap and chart images ===== */
.result-image img {
    border-radius: 10px !important;
    border: 1px solid var(--img-border) !important;
}

/* ===== Accordion panels ===== */
.gr-accordion {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    background: var(--accordion-bg) !important;
    margin-top: 0.6rem !important;
}
.gr-accordion summary {
    color: var(--accordion-label) !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 0.9rem 1.2rem !important;
}

/* ===== Preprocessing step panels ===== */
.step-panel {
    border: 1px solid var(--step-border) !important;
    border-radius: 10px !important;
    background: var(--step-bg) !important;
    overflow: hidden;
}
.step-panel img {
    border-radius: 8px !important;
}

/* ===== VLM button ===== */
#btn-vlm {
    background: var(--vlm-bg) !important;
    border: 1px solid var(--vlm-border) !important;
    color: var(--vlm-text) !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    border-radius: 10px !important;
    height: 46px !important;
    transition: all 0.2s ease !important;
}
#btn-vlm:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--vlm-text) !important;
}

/* ===== VLM output textbox ===== */
#vlm-output textarea {
    background: var(--vlm-input-bg) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--vlm-input-text) !important;
    font-size: 0.93rem !important;
    line-height: 1.65 !important;
}

/* ===== Disclaimer banner ===== */
#disclaimer-banner {
    margin-top: 2rem;
    padding: 1.2rem 1.6rem;
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-left: 4px solid var(--warn-border-left);
    border-radius: 10px;
    color: var(--warn-text);
    font-size: 0.83rem;
    line-height: 1.75;
}
#disclaimer-banner strong {
    color: var(--warn-strong) !important;
}
#disclaimer-banner .copyright {
    margin-top: 0.5rem;
    color: var(--warn-copy);
    font-size: 0.80rem;
}

/* ===== Helper text styling ===== */
.helper-text {
    color: var(--text-helper) !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
}

/* ===== Column equal height ===== */
.equal-height > .gr-column {
    display: flex !important;
    flex-direction: column !important;
}

/* ===== Sample images section ===== */
.examples-section .gallery {
    gap: 8px !important;
}

/* ===== Workflow guide badges ===== */
.workflow-step {
    display: inline-block;
    background: var(--workflow-bg);
    border: 1px solid var(--workflow-border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    color: var(--workflow-text);
    font-size: 0.82rem;
    text-align: center;
    line-height: 1.5;
    flex: 1;
}
.workflow-step strong {
    display: block;
    color: var(--workflow-strong) !important;
    font-size: 0.88rem;
    margin-bottom: 0.15rem;
}
"""


# =============================================================================
# Pre-download VLM weights at startup (runs during Space "Building" phase)
# =============================================================================
# On HF Spaces, the reverse proxy drops idle WebSocket connections after
# roughly 60 seconds.  Downloading 8+ GB of model weights during a button
# click almost always exceeds that limit.  By pre-downloading at import
# time, the files land in the local HF cache before the app starts
# accepting requests.  The subsequent _load_vlm() call then loads from
# disk, which is fast enough to stay within the proxy timeout.

def _predownload_vlm_weights():
    """
    Pre-download VLM model files to the local HF cache at startup.

    This is a DOWNLOAD-ONLY step -- the model is NOT loaded into RAM,
    so it has zero memory impact.  The CNN and VLM together exceed
    16 GB when both are in memory, so we can NOT eagerly load the VLM.

    By caching the files on disk, the subsequent _load_vlm() call
    (triggered by the user clicking the button) loads from local disk
    instead of downloading 8+ GB over the network -- this is fast
    enough for the 10-second heartbeat mechanism to keep the
    connection alive.
    """
    try:
        from huggingface_hub import snapshot_download
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        print("STARTUP  Pre-downloading VLM base model weights (disk only, no RAM) ...")
        snapshot_download(
            HF_MODEL_ID,
            token=hf_token,
            ignore_patterns=["*.md", "*.txt"],
        )
        print("STARTUP  VLM model files cached on disk successfully.")
    except Exception as exc:
        # Non-fatal: the model will be downloaded on first click
        print(f"STARTUP  VLM pre-download skipped: {exc}")

_predownload_vlm_weights()

# =============================================================================
# Gradio Blocks layout
# =============================================================================
# The interface is organised top-to-bottom in the order a user naturally
# interacts with it:
#
#   1. Header with title and brief description
#   2. Three-step workflow guide (Upload -> Analyse -> Review)
#   3. Two-column layout: image upload (left) + prediction results (right)
#   4. Full-width "Run Analysis" button
#   5. Collapsible preprocessing pipeline gallery
#   6. Collapsible VLM clinical explanation panel
#   7. Research disclaimer with copyright notice
#
# This top-to-bottom flow is more intuitive than cramming everything into
# a single viewport.  Each section is visually distinct so users always
# know where they are in the process.
with gr.Blocks(css=CSS, title="SkinCancerVisionXAI", theme=gr.themes.Base()) as demo:

    # -----------------------------------------------------------------
    # Hidden state variables (not rendered; pass data between callbacks)
    # -----------------------------------------------------------------
    pred_label_state = gr.State(None)
    gradcam_state    = gr.State(None)

    # -----------------------------------------------------------------
    # 1. Application header
    # -----------------------------------------------------------------
    gr.HTML("""
    <div id="app-header">
        <h1>SkinCancerVisionXAI</h1>
        <p class="subtitle">
            CBAM-Xception Classifier &nbsp;|&nbsp;
            Grad-CAM++ Attention Maps &nbsp;|&nbsp;
            Optional VLM Clinical Explanation
        </p>
        <div class="divider"></div>
    </div>
    """)

    # Theme toggle button (top-right corner via CSS)
    theme_btn = gr.Button(
        "Light Mode",
        elem_id="theme-toggle",
        size="sm",
    )

    # -----------------------------------------------------------------
    # 2. Three-step workflow guide
    # -----------------------------------------------------------------
    # gr.HTML("""
    # <div style="display:flex; gap:12px; margin:0 0 1.2rem; padding:0 4px;">
    #     <div class="workflow-step">
    #         <strong>Step 1</strong>
    #         Upload or select a dermoscopy image
    #     </div>
    #     <div class="workflow-step">
    #         <strong>Step 2</strong>
    #         Click "Run Analysis" to classify
    #     </div>
    #     <div class="workflow-step">
    #         <strong>Step 3</strong>
    #         Review the prediction and heatmap
    #     </div>
    # </div>
    # """)

    # -----------------------------------------------------------------
    # 3. Main two-column layout
    # -----------------------------------------------------------------
    with gr.Row(equal_height=True, elem_classes=["equal-height"]):

        # -- Left column: image upload + sample images --
        with gr.Column(scale=1, min_width=340):

            gr.Markdown(
                "<p class='section-title'>Input Image</p>"
            )

            input_image = gr.Image(
                label="Skin Lesion Image",
                type="pil",
                sources=["upload", "clipboard"],
                height=340,
                elem_id="upload-section",
            )

            with gr.Accordion(
                "Sample Images  (click any thumbnail to load it)",
                open=True,
                elem_classes=["examples-section"],
            ):
                gr.Markdown(
                    "<p class='helper-text'>"
                    "These sample images are drawn from the HAM10000 dataset. "
                    "Click a thumbnail to load it into the input panel above."
                    "</p>"
                )
                gr.Examples(
                    examples=EXAMPLE_PATHS,
                    inputs=[input_image],
                    label=None,
                )

        # -- Right column: prediction results --
        with gr.Column(scale=1, min_width=340):

            gr.Markdown(
                "<p class='section-title'>Classification Result</p>"
            )

            pred_md = gr.Markdown(
                "*Upload an image and click* ***Run Analysis*** *to see results here.*",
                elem_id="pred-md",
            )

            gradcam_img = gr.Image(
                label="Attention Heatmap  (Grad-CAM++)",
                height=280,
                elem_classes=["result-image"],
            )

            bar_chart = gr.Image(
                label="Prediction Confidence by Class",
                height=260,
                elem_classes=["result-image"],
            )

    # -----------------------------------------------------------------
    # 4. Run Analysis button (full width, prominent)
    # -----------------------------------------------------------------
    btn_analyze = gr.Button(
        "Run Analysis",
        elem_id="btn-analyze",
        size="lg",
    )

    # -----------------------------------------------------------------
    # 5. Preprocessing pipeline gallery (collapsed by default)
    # -----------------------------------------------------------------
    with gr.Accordion("Preprocessing Pipeline  (click to expand)", open=False):
        gr.Markdown(
            "<p class='helper-text'>"
            "Each input image passes through a six-stage preprocessing pipeline "
            "before entering the classifier.  These stages exactly replicate the "
            "offline training pipeline used in the original study, ensuring "
            "consistent model behaviour across training and inference."
            "</p>"
        )
        with gr.Row():
            preproc_orig  = gr.Image(
                label="1.  Original Image",
                height=190,
                elem_classes=["step-panel"],
            )
            preproc_hair  = gr.Image(
                label="2.  Hair Removal",
                height=190,
                elem_classes=["step-panel"],
            )
            preproc_illum = gr.Image(
                label="3.  Illumination Correction",
                height=190,
                elem_classes=["step-panel"],
            )
        with gr.Row():
            preproc_clahe   = gr.Image(
                label="4.  CLAHE Enhancement",
                height=190,
                elem_classes=["step-panel"],
            )
            preproc_resized = gr.Image(
                label=f"5.  Resized ({IMG_H} x {IMG_W})",
                height=190,
                elem_classes=["step-panel"],
            )
            preproc_norm    = gr.Image(
                label="6.  Normalised Input",
                height=190,
                elem_classes=["step-panel"],
            )

    # -----------------------------------------------------------------
    # 6. VLM clinical explanation (collapsed by default)
    # -----------------------------------------------------------------
    with gr.Accordion(
        "VLM Clinical Explanation  (optional -- GPU recommended)", open=False
    ):
        gr.Markdown(
            "<p class='helper-text'>"
            "MedGemma-4B with a fine-tuned LoRA adapter analyses the "
            "Grad-CAM++ attention heatmap and generates a short explanation "
            "of which visual features the model focused on to reach its "
            "prediction.  A GPU is recommended for fast results.  "
            "On CPU, generating an explanation can take 5-10 minutes."
            "</p>"
        )
        btn_vlm = gr.Button(
            "Generate VLM Explanation",
            elem_id="btn-vlm",
        )
        vlm_output = gr.Textbox(
            label="Clinical Explanation",
            lines=5,
            placeholder="The VLM explanation will appear here after running analysis ...",
            interactive=False,
            elem_id="vlm-output",
        )

    # -----------------------------------------------------------------
    # 7. Research disclaimer and copyright
    # -----------------------------------------------------------------
    gr.HTML("""
    <div id="disclaimer-banner">
        <strong>\u26A0 Research Prototype Only.</strong>
        This application is not intended for clinical diagnosis or medical
        decision-making.  The model was fine-tuned on a limited dataset and
        may not generalise to all skin types, lighting conditions, or lesion
        presentations.
        <strong>Always consult a qualified dermatologist.</strong>
        <div class="copyright">
            \u00A9 2026 Faysal Ahmmed, Ajmy Alaly, Samanta Mehnaj,
            Asef Rahman, F.M. Mridha.
        </div>
    </div>
    """)

    # -----------------------------------------------------------------
    # Event bindings
    # -----------------------------------------------------------------

    # --- Theme toggle (client-side only, no server round-trip) ---
    TOGGLE_JS = """
    () => {
        const c = document.querySelector('.gradio-container') || document.body;
        const isLight = c.classList.contains('light-mode');
        if (isLight) {
            c.classList.remove('light-mode');
            try { localStorage.setItem('skinvision-theme', 'dark'); } catch(e) {}
        } else {
            c.classList.add('light-mode');
            try { localStorage.setItem('skinvision-theme', 'light'); } catch(e) {}
        }
        const btn = document.getElementById('theme-toggle');
        if (btn) {
            const inner = btn.querySelector('button') || btn;
            inner.textContent = isLight ? 'Dark Mode' : 'Light Mode';
        }
    }
    """
    theme_btn.click(fn=None, inputs=[], outputs=[], js=TOGGLE_JS)

    # --- Auto-detect theme on page load ---
    INIT_THEME_JS = """
    () => {
        let saved = null;
        try { saved = localStorage.getItem('skinvision-theme'); } catch(e) {}
        const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
        const useLight = saved ? (saved === 'light') : prefersLight;
        const c = document.querySelector('.gradio-container') || document.body;
        if (useLight) {
            c.classList.add('light-mode');
        }
        const btn = document.getElementById('theme-toggle');
        if (btn) {
            const inner = btn.querySelector('button') || btn;
            inner.textContent = useLight ? 'Light Mode' : 'Dark Mode';
        }
    }
    """
    demo.load(fn=None, inputs=[], outputs=[], js=INIT_THEME_JS)

    # --- Wire the "Run Analysis" button to the inference pipeline ---
    # The outputs list must match the order of values returned by
    # run_analysis() exactly.
    btn_analyze.click(
        fn=run_analysis,
        inputs=[input_image],
        outputs=[
            preproc_orig,
            preproc_hair,
            preproc_illum,
            preproc_clahe,
            preproc_resized,
            preproc_norm,
            gradcam_img,
            pred_md,
            bar_chart,
            pred_label_state,
            gradcam_state,
        ],
    )

    # Wire the VLM button.  It consumes the state values produced
    # by run_analysis() and writes into the explanation textbox.
    btn_vlm.click(
        fn=run_vlm,
        inputs=[gradcam_state, pred_label_state],
        outputs=[vlm_output],
    )


# =============================================================================
# Queue configuration (must run at import time for HF Spaces compatibility)
# =============================================================================
# HF Spaces imports this module rather than running it as __main__, so
# demo.queue() must be called at module level.  status_update_rate sends
# periodic heartbeats to keep the WebSocket alive during slow CPU-based
# VLM inference (which can take 10-15 minutes on a CPU-only deployment).

demo.queue(
    status_update_rate=10,
    default_concurrency_limit=1,
)


# =============================================================================
# Entry point (local development only)
# =============================================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        ssr_mode=False,
    )