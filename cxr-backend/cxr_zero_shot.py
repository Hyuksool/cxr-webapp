"""
Zero-shot CXR classification using CLIP (CheXzero-style).

Approach: Compare image embeddings with medical text prompts describing each
pathology. Based on the CheXzero paper (MICCAI 2022 / Nature Biomedical Eng.):
  "Zero-shot Learning of Chest X-ray Diagnosis via Contrastive Visual-Language
   Pretraining" — Tiu et al.

Uses openai/clip-vit-base-patch32 from HuggingFace. For best results use
fine-tuned CheXzero checkpoint when available.

Prompt engineering improvements:
- 3 prompts per pathology (better coverage of radiological descriptions)
- More specific medical language aligned with radiology reports
- Prompts derived from radiology textbooks and CheXpert/MIMIC reporting style

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
# Pathology prompts (CheXzero-style, improved)
# 3 prompts per pathology for better coverage.
# Derived from radiology reporting standards and CheXpert labeling guidelines.
# ─────────────────────────────────────────────
ZERO_SHOT_PATHOLOGIES: dict[str, list[str]] = {
    "Pneumonia": [
        "chest x-ray showing lobar consolidation consistent with bacterial pneumonia",
        "focal airspace opacity with air bronchograms indicating pneumonia",
        "bilateral lower lobe infiltrates with fever consistent with pneumonia",
    ],
    "Pleural Effusion": [
        "blunting of costophrenic angle indicating pleural effusion on chest radiograph",
        "homogeneous opacity in hemithorax with meniscus sign — pleural effusion",
        "layering fluid in bilateral pleural spaces on chest x-ray",
    ],
    "Cardiomegaly": [
        "enlarged cardiac silhouette with cardiothoracic ratio greater than 0.5",
        "cardiomegaly on PA chest radiograph with prominent left heart border",
        "globular enlarged heart shadow suggesting cardiomegaly or pericardial effusion",
    ],
    "Pneumothorax": [
        "pneumothorax with visible pleural line and absent lung markings peripherally",
        "collapsed lung with air in pleural space — pneumothorax on chest x-ray",
        "deep sulcus sign or absent costophrenic angle indicating pneumothorax",
    ],
    "Pulmonary Edema": [
        "bilateral perihilar haziness with interstitial markings — pulmonary edema",
        "butterfly or bat-wing pattern of pulmonary edema on chest radiograph",
        "Kerley B lines and vascular redistribution indicating cardiogenic pulmonary edema",
    ],
    "Atelectasis": [
        "linear or subsegmental atelectasis with crowded pulmonary vessels",
        "plate-like atelectasis at lung base with minor volume loss",
        "lobar collapse with ipsilateral mediastinal shift — obstructive atelectasis",
    ],
    "Consolidation": [
        "homogeneous airspace consolidation with air bronchograms",
        "lobar or segmental consolidation obscuring the cardiac border — silhouette sign",
        "dense airspace opacity without volume loss — consolidation on chest x-ray",
    ],
    "Lung Nodule": [
        "solitary pulmonary nodule less than 3 cm on chest x-ray",
        "well-defined round opacity in lung parenchyma — pulmonary nodule",
        "calcified or non-calcified nodule in lung field on chest radiograph",
    ],
    "Aortic Enlargement": [
        "widened mediastinum greater than 8 cm suggesting aortic pathology",
        "prominent aortic knob and unfolded aorta on chest radiograph",
        "aortic enlargement with mediastinal widening — r/o aortic aneurysm or dissection",
    ],
    "Emphysema": [
        "hyperinflated lungs with flattened diaphragms consistent with emphysema",
        "increased anteroposterior diameter and hyperlucent lung fields — COPD emphysema",
        "bullae and decreased vascular markings bilaterally — emphysematous changes",
    ],
    "Fracture": [
        "rib fracture with cortical discontinuity on chest radiograph",
        "acute rib fractures along lateral chest wall following trauma",
        "multiple rib fractures with associated hemothorax or pneumothorax",
    ],
    "No Finding": [
        "normal chest x-ray with clear lung fields and no acute cardiopulmonary finding",
        "unremarkable chest radiograph without infiltrate, effusion, or pneumothorax",
        "normal PA chest radiograph with no acute pathology identified",
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
    "Emphysema": "routine",
    "Fracture": "routine",
    "No Finding": "normal",
}

ZERO_SHOT_THRESHOLD = 0.22  # Similarity threshold for positive detection (tuned)

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

    For improved accuracy, consider BiomedCLIP:
        model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    (requires: pip install open_clip_torch)
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
    """Pre-compute and cache text embeddings for all pathology prompts.

    Averages embeddings across 3 prompt variants per pathology for
    more robust text representation.
    """
    embeddings = {}
    for pathology, prompts in ZERO_SHOT_PATHOLOGIES.items():
        with torch.no_grad():
            inputs = _CLIP_PROCESSOR(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,  # CLIP max token length
            ).to(_DEVICE)
            text_feats = _CLIP_MODEL.get_text_features(**inputs)
            # Newer transformers may return ModelOutput instead of tensor
            if not isinstance(text_feats, torch.Tensor):
                text_feats = text_feats.pooler_output if hasattr(text_feats, "pooler_output") else text_feats[0]
            # Normalize each prompt embedding before averaging (better representation)
            text_feats_norm = text_feats / text_feats.norm(dim=-1, keepdim=True)
            # Average normalized embeddings across prompts
            text_feats_avg = text_feats_norm.mean(dim=0, keepdim=True)
            # Re-normalize the averaged embedding
            text_feats_avg = text_feats_avg / text_feats_avg.norm(dim=-1, keepdim=True)
            embeddings[pathology] = text_feats_avg
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
            if not isinstance(image_feats, torch.Tensor):
                image_feats = image_feats.pooler_output if hasattr(image_feats, "pooler_output") else image_feats[0]
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
            "model": "openai/clip-vit-base-patch32 (zero-shot, improved prompts)",
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
