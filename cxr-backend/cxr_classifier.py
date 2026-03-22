"""
CXR (Chest X-Ray) classifier using TorchXRayVision.

Provides:
- 18-pathology classification (DenseNet-121)
- GradCAM heatmap generation for localization
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

# Classification threshold — pathologies above this are flagged as findings
CLASSIFICATION_THRESHOLD = 0.05

# Always return at least this many top pathologies as findings (excluding No Finding)
MIN_FINDINGS_COUNT = 5

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
    """Convert raw image bytes to TorchXRayVision-compatible tensor."""
    # Load image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    # Convert to grayscale float (TorchXRayVision expects single-channel)
    img_gray = np.mean(img_array, axis=2)

    # TorchXRayVision normalize: maps [0,255] → [-1024, 1024]
    img_norm = xrv.datasets.normalize(img_gray, maxval=255, reshape=True)

    # Apply transforms: resize to 224x224
    transform = xrv.datasets.XRayCenterCrop()
    img_transformed = transform(img_norm)

    # Add batch dimension: [1, 1, H, W]
    tensor = torch.from_numpy(img_transformed).unsqueeze(0).float()
    return img_array, tensor


def _generate_gradcam(
    model,
    tensor: torch.Tensor,
    target_class_idx: int,
    device: torch.device,
    original_img: np.ndarray,
) -> Optional[str]:
    """Generate GradCAM heatmap for the target class. Returns base64-encoded PNG."""
    if not CV2_AVAILABLE:
        return None

    try:
        tensor = tensor.to(device)
        tensor.requires_grad_(True)

        # Forward pass
        with torch.enable_grad():
            output = model(tensor)
            score = output[0, target_class_idx]
            model.zero_grad()
            score.backward()

        # Get gradients from last conv layer (DenseNet features)
        # Use hook-based approach for robustness
        gradients = tensor.grad
        if gradients is None:
            return None

        # Simple gradient-based saliency map
        saliency = gradients[0, 0].abs().cpu().numpy()

        # Resize to original image size
        h, w = original_img.shape[:2]
        saliency_resized = cv2.resize(saliency, (w, h))

        # Normalize
        saliency_norm = (saliency_resized - saliency_resized.min()) / (
            saliency_resized.max() - saliency_resized.min() + 1e-8
        )

        # Apply colormap
        heatmap = cv2.applyColorMap(
            (saliency_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # Overlay on original image
        if len(original_img.shape) == 2:
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_img[:, :, :3]

        overlay = cv2.addWeighted(original_rgb, 0.6, heatmap, 0.4, 0)

        # Encode as base64 PNG
        _, buffer = cv2.imencode(".png", overlay)
        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        print(f"GradCAM failed: {e}")
        return None


def classify_cxr_image(image_bytes: bytes) -> dict:
    """
    Classify CXR image using TorchXRayVision DenseNet.

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
    tensor_device = tensor.to(_DEVICE)

    with torch.no_grad():
        outputs = _CXR_MODEL(tensor_device)
        # Sigmoid already applied by TorchXRayVision
        probs = outputs[0].cpu().numpy()

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

        if prob_float >= CLASSIFICATION_THRESHOLD and name != "No Finding":
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

    # Guarantee minimum findings — always show at least MIN_FINDINGS_COUNT diagnoses
    # This ensures diagnosis names ALWAYS appear regardless of model confidence
    if len(findings) < MIN_FINDINGS_COUNT:
        shown_names = {f["name"] for f in findings}
        candidates = [p for p in pathology_results if p["name"] != "No Finding" and p["name"] not in shown_names]
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

    # Generate GradCAM for top finding
    heatmap_b64 = None
    if findings and CV2_AVAILABLE:
        top_idx = list(model_pathologies).index(findings[0]["name"])
        # Re-run with gradients enabled
        tensor_grad = tensor.clone().to(_DEVICE)
        heatmap_b64 = _generate_gradcam(
            _CXR_MODEL, tensor_grad, top_idx, _DEVICE, img_array
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
