"""
Zero-shot CXR classification using CLIP (CheXzero-style).

Approach: Compare image embeddings with medical text prompts describing each
pathology. Based on the CheXzero paper (MICCAI 2022 / Nature Biomedical Eng.):
  "Zero-shot Learning of Chest X-ray Diagnosis via Contrastive Visual-Language
   Pretraining" — Tiu et al.

Uses openai/clip-vit-base-patch32 from HuggingFace. For best results use
fine-tuned CheXzero checkpoint when available.

Usage:
    from cxr_zero_shot import classify_zero_shot, preload_zero_shot_model
"""

import io
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image

# ─────────────────────────────────────────────
# Optional imports — fail gracefully
# ─────────────────────────────────────────────
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("WARNING: transformers not installed. Zero-shot CXR disabled.")
    print("         Install with: pip install transformers")

# ─────────────────────────────────────────────
# Pathology prompts (CheXzero-style paired prompts)
# Each entry: [positive prompt, negative/normal comparison]
# ─────────────────────────────────────────────
ZERO_SHOT_PATHOLOGIES: dict[str, list[str]] = {
    "Pneumonia": [
        "chest x-ray showing pneumonia with consolidation",
        "bilateral infiltrates consistent with pneumonia",
    ],
    "Pleural Effusion": [
        "chest x-ray showing pleural effusion with blunting of costophrenic angles",
        "fluid in the pleural space on chest radiograph",
    ],
    "Cardiomegaly": [
        "enlarged heart on chest x-ray, cardiomegaly",
        "cardiothoracic ratio greater than 0.5 on chest radiograph",
    ],
    "Pneumothorax": [
        "pneumothorax on chest x-ray with collapsed lung",
        "absence of lung markings indicating pneumothorax",
    ],
    "Pulmonary Edema": [
        "pulmonary edema on chest x-ray with bilateral perihilar haziness",
        "interstitial and alveolar edema on chest radiograph",
    ],
    "Atelectasis": [
        "atelectasis on chest x-ray with linear opacities",
        "partial lung collapse visible on radiograph",
    ],
    "Consolidation": [
        "consolidation in lung on chest x-ray",
        "airspace opacity consistent with consolidation",
    ],
    "Lung Nodule": [
        "pulmonary nodule or mass on chest x-ray",
        "solitary lung nodule visible on chest radiograph",
    ],
    "Aortic Enlargement": [
        "widened mediastinum on chest x-ray",
        "enlarged aortic knob on chest radiograph",
    ],
    "No Finding": [
        "normal chest x-ray without significant abnormality",
        "clear lungs with no acute cardiopulmonary process",
    ],
}

# Urgency per zero-shot pathology
ZERO_SHOT_URGENCY: dict[str, str] = {
    "Pneumothorax": "critical",
    "Pneumonia": "urgent",
    "Pleural Effusion": "urgent",
    "Pulmonary Edema": "urgent",
    "Cardiomegaly": "routine",
    "Atelectasis": "routine",
    "Consolidation": "urgent",
    "Lung Nodule": "urgent",
    "Aortic Enlargement": "routine",
    "No Finding": "normal",
}

ZERO_SHOT_THRESHOLD = 0.25  # Similarity threshold for positive detection

# ─────────────────────────────────────────────
# Global model cache
# ─────────────────────────────────────────────
_CLIP_MODEL: Optional[object] = None
_CLIP_PROCESSOR: Optional[object] = None
_DEVICE: Optional[torch.device] = None

# Pre-computed text embeddings cache
_TEXT_EMBEDDINGS: Optional[dict] = None


def preload_zero_shot_model(model_name: str = "openai/clip-vit-base-patch32") -> bool:
    """
    Load CLIP model for zero-shot classification.

    Downloads ~600MB on first call; cached in ~/.cache/huggingface afterward.
    Set ENABLE_ZERO_SHOT=false to disable.
    """
    global _CLIP_MODEL, _CLIP_PROCESSOR, _DEVICE, _TEXT_EMBEDDINGS

    if os.environ.get("ENABLE_ZERO_SHOT", "true").lower() == "false":
        print("Zero-shot CXR disabled via ENABLE_ZERO_SHOT=false")
        return False

    if not CLIP_AVAILABLE:
        print("CLIP not available (transformers not installed)")
        return False

    if _CLIP_MODEL is not None:
        return True  # Already loaded

    # Device selection (CPU preferred for Railway)
    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
    else:
        _DEVICE = torch.device("cpu")

    print(f"Loading CLIP model '{model_name}' on {_DEVICE} for zero-shot CXR...")

    try:
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
        _CLIP_MODEL = CLIPModel.from_pretrained(model_name).to(_DEVICE)
        _CLIP_MODEL.eval()

        # Pre-compute text embeddings for all pathologies
        _TEXT_EMBEDDINGS = _precompute_text_embeddings()
        print(f"Zero-shot model loaded. Pathologies: {list(ZERO_SHOT_PATHOLOGIES.keys())}")
        return True

    except Exception as e:
        print(f"ERROR loading CLIP model: {e}")
        _CLIP_MODEL = None
        _CLIP_PROCESSOR = None
        return False


def _precompute_text_embeddings() -> dict:
    """Pre-compute and cache text embeddings for all pathology prompts."""
    embeddings = {}
    for pathology, prompts in ZERO_SHOT_PATHOLOGIES.items():
        with torch.no_grad():
            inputs = _CLIP_PROCESSOR(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(_DEVICE)
            text_feats = _CLIP_MODEL.get_text_features(**inputs)
            # Average over prompt variants
            text_feats = text_feats.mean(dim=0, keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            embeddings[pathology] = text_feats
    return embeddings


def classify_zero_shot(image_bytes: bytes) -> dict:
    """
    Classify CXR image using CLIP zero-shot approach.

    Returns:
        dict with keys:
            - available: bool (False if model not loaded)
            - pathologies: list of {name, similarity, urgency} sorted by similarity
            - findings: list above threshold
            - model: str describing the model used
    """
    if _CLIP_MODEL is None or _TEXT_EMBEDDINGS is None:
        return {
            "available": False,
            "pathologies": [],
            "findings": [],
            "model": "zero-shot unavailable",
        }

    try:
        # Load and process image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        with torch.no_grad():
            inputs = _CLIP_PROCESSOR(
                images=img,
                return_tensors="pt",
            ).to(_DEVICE)
            image_feats = _CLIP_MODEL.get_image_features(**inputs)
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

        # Compute similarity with each pathology
        pathology_scores = []
        for pathology, text_emb in _TEXT_EMBEDDINGS.items():
            similarity = float((image_feats @ text_emb.T).squeeze())
            # Normalize to [0, 1] range (cosine similarity is [-1, 1])
            similarity_norm = (similarity + 1) / 2
            urgency = ZERO_SHOT_URGENCY.get(pathology, "routine")

            pathology_scores.append({
                "name": pathology,
                "similarity": round(similarity_norm, 4),
                "urgency": urgency,
            })

        # Sort by similarity descending
        pathology_scores.sort(key=lambda x: x["similarity"], reverse=True)

        # Apply threshold — exclude "No Finding" from positive findings
        findings = [
            p for p in pathology_scores
            if p["similarity"] >= ZERO_SHOT_THRESHOLD and p["name"] != "No Finding"
        ]

        return {
            "available": True,
            "pathologies": pathology_scores,
            "findings": findings,
            "model": "openai/clip-vit-base-patch32 (zero-shot)",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "available": False,
            "pathologies": [],
            "findings": [],
            "model": f"zero-shot error: {e}",
        }
