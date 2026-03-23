"""
CXR (Chest X-Ray) classifier using TorchXRayVision.

Provides:
- 18-pathology classification (DenseNet-121)
- GradCAM heatmap generation for localization (proper hook-based GradCAM)
- Test-Time Augmentation (TTA) for improved accuracy
- CLAHE preprocessing for contrast enhancement
- Urgency level determination

Usage:
    from cxr_classifier import classify_cxr_image, preload_cxr_model
"""

import io
import base64
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import torchxrayvision as xrv
    XRV_AVAILABLE = True
except ImportError:
    XRV_AVAILABLE = False
    print("WARNING: torchxrayvision not installed. Run: pip install torchxrayvision")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# TorchXRayVision pathology labels (DenseNet models)
PATHOLOGY_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
    "Infiltration",
    "Emphysema",
    "Fibrosis",
    "Hernia",
]

# Urgency mapping per pathology
URGENCY_MAP = {
    # Critical — immediate intervention required
    "Pneumothorax": "critical",
    # Urgent — requires prompt evaluation
    "Consolidation": "urgent",
    "Pneumonia": "urgent",
    "Edema": "urgent",
    "Effusion": "urgent",          # actual label in densenet121-res224-all
    "Pleural Effusion": "urgent",  # alias
    "Lung Lesion": "urgent",
    "Mass": "urgent",
    "Nodule": "urgent",
    # Routine — needs follow-up
    "Cardiomegaly": "routine",
    "Atelectasis": "routine",
    "Enlarged Cardiomediastinum": "routine",
    "Lung Opacity": "routine",
    "Fracture": "routine",
    "Infiltration": "routine",
    "Fibrosis": "routine",
    "Emphysema": "routine",
    "Pleural_Thickening": "routine",
    "Pleural Other": "routine",
    "Hernia": "routine",
    # Info only
    "Support Devices": "info",
    # Normal
    "No Finding": "normal",
}

# Per-pathology detection thresholds (calibrated for sensitivity/specificity balance)
# Critical pathologies use lower thresholds to maximize sensitivity
PATHOLOGY_THRESHOLDS: dict[str, float] = {
    "Pneumothorax": 0.08,          # critical — maximize sensitivity
    "Enlarged Cardiomediastinum": 0.10,  # critical when high
    "Pneumonia": 0.15,
    "Consolidation": 0.15,
    "Edema": 0.15,
    "Pleural Effusion": 0.15,
    "Effusion": 0.15,
    "Lung Lesion": 0.12,
    "Atelectasis": 0.20,
    "Cardiomegaly": 0.20,
    "Infiltration": 0.20,
    "Fracture": 0.15,
    "Emphysema": 0.20,
    "Fibrosis": 0.20,
    "Lung Opacity": 0.20,
    "Pleural Other": 0.15,
    "Hernia": 0.15,
    "Support Devices": 0.30,
    "No Finding": 0.50,
}
DEFAULT_THRESHOLD = 0.15

# Minimum probability for supplementary findings (below per-pathology threshold)
MIN_SUPPLEMENTARY_THRESHOLD = 0.08

# Always return at least this many top pathologies as findings (excluding No Finding)
MIN_FINDINGS_COUNT = 3

# Global model cache
_CXR_MODEL: Optional[object] = None
_DEVICE: Optional[torch.device] = None


def preload_cxr_model():
    """Load TorchXRayVision DenseNet model into memory. Call at startup."""
    global _CXR_MODEL, _DEVICE

    if not XRV_AVAILABLE:
        print("TorchXRayVision not available — CXR classification disabled")
        return False

    if torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
    else:
        _DEVICE = torch.device("cpu")

    print(f"Loading TorchXRayVision DenseNet on device: {_DEVICE}")

    try:
        # densenet121-res224-all: trained on all available datasets (best coverage)
        _CXR_MODEL = xrv.models.DenseNet(weights="densenet121-res224-all")
        _CXR_MODEL = _CXR_MODEL.to(_DEVICE)
        _CXR_MODEL.eval()
        print(f"CXR model loaded. Pathologies: {_CXR_MODEL.pathologies}")
        return True
    except Exception as e:
        print(f"ERROR loading CXR model: {e}")
        return False


def _preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, torch.Tensor]:
    """Convert raw image bytes to TorchXRayVision-compatible tensor.

    Improvements over original:
    - Luminance-weighted grayscale (ITU-R BT.709) instead of channel mean
    - CLAHE contrast enhancement (if cv2 available)
    """
    # Load image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    # Luminance-weighted grayscale (ITU-R BT.709) — more accurate than channel mean
    # Weights: R=0.2126, G=0.7152, B=0.0722
    r = img_array[:, :, 0].astype(np.float32)
    g = img_array[:, :, 1].astype(np.float32)
    b = img_array[:, :, 2].astype(np.float32)
    img_gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # CLAHE contrast enhancement — improves detection of subtle findings
    # Applied before TorchXRayVision normalization
    if CV2_AVAILABLE:
        img_uint8 = np.clip(img_gray, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_uint8).astype(np.float32)

    # TorchXRayVision normalize: maps [0,255] → [-1024, 1024]
    img_norm = xrv.datasets.normalize(img_gray, maxval=255, reshape=True)

    # Apply transforms: resize to 224x224
    transform = xrv.datasets.XRayCenterCrop()
    img_transformed = transform(img_norm)

    # Add batch dimension: [1, 1, H, W]
    tensor = torch.from_numpy(img_transformed).unsqueeze(0).float()
    return img_array, tensor


def _run_inference(tensor: torch.Tensor) -> np.ndarray:
    """Run model inference on a tensor and return probabilities.

    Uses TTA (Test-Time Augmentation): averages predictions from
    original image and horizontally flipped version.
    Horizontal flip is valid for CXR: lung pathologies are bilateral,
    and averaging reduces model bias.
    """
    with torch.no_grad():
        tensor_device = tensor.to(_DEVICE)
        probs_orig = _CXR_MODEL(tensor_device)[0].cpu().numpy()

        # Horizontal flip TTA
        tensor_flipped = torch.flip(tensor_device, dims=[3])
        probs_flipped = _CXR_MODEL(tensor_flipped)[0].cpu().numpy()

    # Average original + flipped predictions
    return (probs_orig + probs_flipped) / 2.0


def _generate_gradcam(
    model,
    tensor: torch.Tensor,
    target_class_idx: int,
    device: torch.device,
    original_img: np.ndarray,
) -> Optional[str]:
    """Proper GradCAM using activation/gradient hooks on last DenseNet denseblock.

    Uses denseblock4 (last feature block) for spatial localization.
    Much more accurate than input-gradient saliency maps.
    """
    if not CV2_AVAILABLE:
        return None

    activations_store: dict = {}
    gradients_store: dict = {}

    def forward_hook(module, input, output):
        activations_store['act'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients_store['grad'] = grad_output[0].detach()

    # Hook on last denseblock (denseblock4) — final spatial feature maps
    try:
        target_layer = model.features.denseblock4
    except AttributeError:
        # Fallback: try last layer of features
        try:
            target_layer = list(model.features.children())[-2]
        except Exception:
            return None

    fh = target_layer.register_forward_hook(forward_hook)
    # register_full_backward_hook available since PyTorch 1.8
    try:
        bh = target_layer.register_full_backward_hook(backward_hook)
    except AttributeError:
        bh = target_layer.register_backward_hook(backward_hook)

    try:
        tensor_device = tensor.clone().to(device)

        with torch.enable_grad():
            output = model(tensor_device)
            score = output[0, target_class_idx]
            model.zero_grad()
            score.backward()

        if 'act' not in activations_store or 'grad' not in gradients_store:
            return None

        act = activations_store['act']   # [1, C, H, W]
        grad = gradients_store['grad']   # [1, C, H, W]

        # Global average pooling of gradients (GradCAM weighting)
        weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted sum of activation maps
        cam = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)  # only positive influence matters
        cam = cam.squeeze().cpu().numpy()

        if cam.max() == cam.min():
            return None

        # Normalize to [0, 1]
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Resize to original image dimensions
        h, w = original_img.shape[:2]
        cam_resized = cv2.resize(cam_norm, (w, h))

        # Apply JET colormap
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # Overlay on original image
        if len(original_img.shape) == 2:
            original_rgb = cv2.cvtColor(original_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif original_img.ndim == 3 and original_img.shape[2] == 4:
            original_rgb = original_img[:, :, :3].astype(np.uint8)
        else:
            original_rgb = original_img.astype(np.uint8)

        overlay = cv2.addWeighted(original_rgb, 0.6, heatmap, 0.4, 0)

        # Encode as base64 PNG
        _, buffer = cv2.imencode(".png", overlay)
        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        print(f"GradCAM failed: {e}")
        return None
    finally:
        fh.remove()
        bh.remove()


def classify_cxr_image(image_bytes: bytes) -> dict:
    """
    Classify CXR image using TorchXRayVision DenseNet.

    Improvements:
    - Luminance-weighted grayscale conversion
    - CLAHE contrast enhancement
    - Test-Time Augmentation (TTA) with horizontal flip
    - Per-pathology calibrated thresholds
    - Proper GradCAM with denseblock4 hooks

    Returns:
        dict with keys:
            - pathologies: list of {name, probability, urgency}
            - findings: list of pathologies above threshold
            - urgency_level: 'critical' | 'urgent' | 'routine' | 'normal'
            - heatmap_base64: optional GradCAM overlay PNG
            - model_pathologies: list of all model pathology names
            - confidence_score: max probability among findings
    """
    if not XRV_AVAILABLE or _CXR_MODEL is None:
        raise RuntimeError(
            "CXR model not loaded. Call preload_cxr_model() first."
        )

    img_array, tensor = _preprocess_image(image_bytes)

    # Run inference with TTA (original + horizontally flipped)
    probs = _run_inference(tensor)

    # Build pathology results using model's own pathology labels
    model_pathologies = _CXR_MODEL.pathologies
    pathology_results = []
    findings = []
    urgency_level = "normal"

    for i, (name, prob) in enumerate(zip(model_pathologies, probs)):
        prob_float = float(prob)
        urgency = URGENCY_MAP.get(name, "routine")

        pathology_results.append({
            "name": name,
            "probability": round(prob_float, 4),
            "urgency": urgency,
        })

        # Per-pathology threshold for findings
        threshold = PATHOLOGY_THRESHOLDS.get(name, DEFAULT_THRESHOLD)
        if prob_float >= threshold and name != "No Finding":
            findings.append({
                "name": name,
                "probability": round(prob_float, 4),
                "urgency": urgency,
            })

            # Escalate urgency level (use 0.2 threshold for urgency decisions)
            if prob_float >= 0.2:
                if urgency == "critical":
                    urgency_level = "critical"
                elif urgency == "urgent" and urgency_level not in ("critical",):
                    urgency_level = "urgent"
                elif urgency == "routine" and urgency_level == "normal":
                    urgency_level = "routine"

    # Sort findings by probability descending
    findings.sort(key=lambda x: x["probability"], reverse=True)
    pathology_results.sort(key=lambda x: x["probability"], reverse=True)

    # Guarantee minimum findings — show at least MIN_FINDINGS_COUNT diagnoses
    # But only include findings above a minimum meaningful threshold
    if len(findings) < MIN_FINDINGS_COUNT:
        shown_names = {f["name"] for f in findings}
        candidates = [
            p for p in pathology_results
            if p["name"] != "No Finding"
            and p["name"] not in shown_names
            and p["probability"] >= MIN_SUPPLEMENTARY_THRESHOLD
        ]
        for p in candidates:
            if len(findings) >= MIN_FINDINGS_COUNT:
                break
            findings.append(p)
        findings.sort(key=lambda x: x["probability"], reverse=True)

    # No Finding check
    no_finding_idx = list(model_pathologies).index("No Finding") if "No Finding" in model_pathologies else -1
    no_finding_prob = float(probs[no_finding_idx]) if no_finding_idx >= 0 else 0.0

    if not findings or no_finding_prob > 0.7:
        urgency_level = "normal"

    # Generate GradCAM for top finding using proper hook-based GradCAM
    heatmap_b64 = None
    if findings:
        top_name = findings[0]["name"]
        if top_name in list(model_pathologies):
            top_idx = list(model_pathologies).index(top_name)
            # GradCAM uses original tensor (no TTA) for clean gradient computation
            heatmap_b64 = _generate_gradcam(
                _CXR_MODEL, tensor, top_idx, _DEVICE, img_array
            )

    confidence = findings[0]["probability"] if findings else no_finding_prob

    return {
        "pathologies": pathology_results,
        "findings": findings,
        "urgency_level": urgency_level,
        "heatmap_base64": heatmap_b64,
        "model_pathologies": list(model_pathologies),
        "confidence_score": round(confidence, 4),
        "no_finding_probability": round(no_finding_prob, 4),
    }
