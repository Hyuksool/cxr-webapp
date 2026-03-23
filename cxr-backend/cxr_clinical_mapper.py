"""
CXR Clinical Diagnosis Mapper (TorchXRayVision 18-label version)

Maps TorchXRayVision DenseNet 18 pathologies to clinical ED/ICU diagnoses,
mirroring the ECG clinical mapper architecture (ecg-webapp/ecg-backend/mimic_clinical_mapper.py).

TorchXRayVision labels (18): Atelectasis, Cardiomegaly, Consolidation, Edema,
Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding,
Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices,
Infiltration, Emphysema, Fibrosis, Hernia

Additional labels vs CheXpert-14: Infiltration, Emphysema, Fibrosis, Hernia
"""

from dataclasses import dataclass
from enum import IntEnum


class ClinicalTier(IntEnum):
    TIER1_IMMEDIATELY_LETHAL = 1
    TIER2_URGENT = 2
    TIER3_IMPORTANT = 3


@dataclass
class CXRClinicalCondition:
    id: int
    name: str
    category: str
    cxr_patterns: list[str]
    icd10_codes: list[str]
    tier: ClinicalTier
    description: str
    min_probability: float = 0.3
    requires_all: bool = False


CXR_CLINICAL_CONDITIONS = [
    # === A. Acute Cardiac (3) ===
    CXRClinicalCondition(
        id=1, name="Acute Heart Failure / Pulmonary Edema",
        category="A. Acute Cardiac",
        cxr_patterns=["Edema", "Cardiomegaly", "Pleural Effusion"],
        icd10_codes=["I50.1", "I50.21", "I50.31", "J81.0"],
        tier=ClinicalTier.TIER1_IMMEDIATELY_LETHAL,
        description="Pulmonary edema with cardiomegaly — acute decompensated heart failure",
        min_probability=0.25,
    ),
    CXRClinicalCondition(
        id=2, name="Cardiomegaly (Chronic Heart Disease)",
        category="A. Acute Cardiac",
        cxr_patterns=["Cardiomegaly"],
        icd10_codes=["I51.7", "I25.10"],
        tier=ClinicalTier.TIER3_IMPORTANT,
        description="Isolated cardiomegaly without acute edema",
    ),
    CXRClinicalCondition(
        id=3, name="Pericardial Effusion / Tamponade",
        category="A. Acute Cardiac",
        cxr_patterns=["Cardiomegaly", "Enlarged Cardiomediastinum"],
        icd10_codes=["I30.9", "I31.9"],
        tier=ClinicalTier.TIER2_URGENT,
        description="Enlarged cardiac silhouette + widened mediastinum — r/o pericardial effusion",
        requires_all=True,
    ),

    # === B. Pulmonary Infection (4) ===
    CXRClinicalCondition(
        id=4, name="Community-Acquired Pneumonia",
        category="B. Pulmonary Infection",
        cxr_patterns=["Pneumonia", "Consolidation", "Infiltration"],
        icd10_codes=["J18.9", "J15.9", "J13"],
        tier=ClinicalTier.TIER2_URGENT,
        description="Focal consolidation/infiltrate consistent with pneumonia",
    ),
    CXRClinicalCondition(
        id=5, name="Pneumonia with Parapneumonic Effusion",
        category="B. Pulmonary Infection",
        cxr_patterns=["Pneumonia", "Pleural Effusion"],
        icd10_codes=["J18.9", "J91.0"],
        tier=ClinicalTier.TIER2_URGENT,
        description="Pneumonia complicated by pleural effusion — consider thoracentesis",
        requires_all=True,
    ),
    CXRClinicalCondition(
        id=6, name="Lung Abscess / Necrotizing Pneumonia",
        category="B. Pulmonary Infection",
        cxr_patterns=["Consolidation", "Lung Lesion"],
        icd10_codes=["J85.1", "J85.2"],
        tier=ClinicalTier.TIER2_URGENT,
        description="Consolidation with cavitary lesion — lung abscess",
        requires_all=True,
    ),
    CXRClinicalCondition(
        id=7, name="Pulmonary Infiltrate (Unspecified)",
        category="B. Pulmonary Infection",
        cxr_patterns=["Infiltration"],
        icd10_codes=["R91.8"],
        tier=ClinicalTier.TIER3_IMPORTANT,
        description="Pulmonary infiltrate — differential includes infection, inflammation, hemorrhage",
    ),

    # === C. Pneumothorax (2) ===
    CXRClinicalCondition(
        id=8, name="Pneumothorax",
        category="C. Pneumothorax",
        cxr_patterns=["Pneumothorax"],
        icd10_codes=["J93.0", "J93.1", "J93.9"],
        tier=ClinicalTier.TIER1_IMMEDIATELY_LETHAL,
        description="Pneumothorax — assess for tension physiology",
    ),
    CXRClinicalCondition(
        id=9, name="Tension Pneumothorax",
        category="C. Pneumothorax",
        cxr_patterns=["Pneumothorax", "Enlarged Cardiomediastinum"],
        icd10_codes=["J93.0"],
        tier=ClinicalTier.TIER1_IMMEDIATELY_LETHAL,
        description="Pneumothorax with mediastinal shift — tension pneumothorax",
        requires_all=True,
    ),

    # === D. Pleural Disease (2) ===
    CXRClinicalCondition(
        id=10, name="Pleural Effusion",
        category="D. Pleural Disease",
        cxr_patterns=["Pleural Effusion"],
        icd10_codes=["J90", "J91.0", "J91.8"],
        tier=ClinicalTier.TIER3_IMPORTANT,
        description="Pleural effusion — consider etiology workup",
    ),
    CXRClinicalCondition(
        id=11, name="Pleural Disease (Other)",
        category="D. Pleural Disease",
        cxr_patterns=["Pleural Other"],
        icd10_codes=["J94.9"],
        tier=ClinicalTier.TIER3_IMPORTANT,
        description="Pleural thickening/calcification or other pleural abnormality",
    ),

    # === E. Pulmonary Mass / Nodule (2) ===
    CXRClinicalCondition(
        id=12, name="Suspected Lung Malignancy",
        category="E. Pulmonary Mass",
        cxr_patterns=["Lung Lesion"],
        icd10_codes=["R91.1", "C34.90"],
        tier=ClinicalTier.TIER2_URGENT,
        description="Focal lung lesion — urgent CT and tissue sampling recommended",
    ),
    CXRClinicalCondition(
        id=13, name="Pulmonary Nodule / Opacity (Incidental)",
        category="E. Pulmonary Mass",
        cxr_patterns=["Lung Opacity"],
        icd10_codes=["R91.1"],
        tier=ClinicalTier.TIER3_IMPORTANT,
        description="Pulmonary opacity — follow Fleischner guidelines if nodular",
        min_probability=0.4,
    ),

    # === F. Atelectasis / Airway (2) ===
    CXRClinicalCondition(
        id=14, name="Atelectasis",
        category="F. Atelectasis",
        cxr_patterns=["Atelectasis"],
        icd10_codes=["J98.11"],
        tier=ClinicalTier.TIER3_IMPORTANT,
        description="Lung collapse — likely subsegmental, may indicate mucus plug or hypoventilation",
    ),
    CXRClinicalCondition(
        id=15, name="Post-Obstructive Collapse",
        category="F. Atelectasis",
        cxr_patterns=["Atelectasis", "Lung Lesion"],
        icd10_codes=["J98.11", "R91.1"],
        tier=ClinicalTier.TIER2_URGENT,
        description="Atelectasis with mass lesion — post-obstructive collapse, urgent bronchoscopy",
        requires_all=True,
    ),

    # === G. ARDS / Diffuse Lung (2) ===
    CXRClinicalCondition(
        id=16, name="ARDS (Acute Respiratory Distress Syndrome)",
        category="G. Diffuse Lung Disease",
        cxr_patterns=["Edema", "Consolidation", "Lung Opacity"],
        icd10_codes=["J80"],
        tier=ClinicalTier.TIER1_IMMEDIATELY_LETHAL,
        description="Bilateral diffuse opacities — ARDS if non-cardiogenic",
        min_probability=0.3,
    ),
    CXRClinicalCondition(
        id=17, name="Interstitial Lung Disease / Fibrosis",
        category="G. Diffuse Lung Disease",
        cxr_patterns=["Fibrosis", "Lung Opacity"],
        icd10_codes=["J84.9", "J84.10"],
        tier=ClinicalTier.TIER3_IMPORTANT,
        description="Interstitial fibrotic pattern — consider HRCT for characterization",
    ),

    # === H. COPD / Emphysema (1) — TorchXRayVision-specific ===
    CXRClinicalCondition(
        id=18, name="COPD / Emphysema",
        category="H. Obstructive Lung Disease",
        cxr_patterns=["Emphysema"],
        icd10_codes=["J43.9", "J44.1"],
        tier=ClinicalTier.TIER3_IMPORTANT,
        description="Hyperinflated lungs with emphysematous changes — correlate with PFTs",
    ),

    # === I. Mediastinal (1) ===
    CXRClinicalCondition(
        id=19, name="Widened Mediastinum (Aortic Pathology)",
        category="I. Mediastinal",
        cxr_patterns=["Enlarged Cardiomediastinum"],
        icd10_codes=["I71.00", "I71.01", "R93.1"],
        tier=ClinicalTier.TIER1_IMMEDIATELY_LETHAL,
        description="Widened mediastinum — r/o aortic dissection/aneurysm, urgent CT angiography",
    ),

    # === J. Diaphragmatic (1) — TorchXRayVision-specific ===
    CXRClinicalCondition(
        id=20, name="Diaphragmatic Hernia",
        category="J. Diaphragmatic",
        cxr_patterns=["Hernia"],
        icd10_codes=["K44.9", "Q79.0"],
        tier=ClinicalTier.TIER2_URGENT,
        description="Diaphragmatic hernia — CT confirmation and surgical consultation recommended",
    ),

    # === K. Support Devices (1) ===
    CXRClinicalCondition(
        id=21, name="Medical Device Positioning",
        category="K. Support Devices",
        cxr_patterns=["Support Devices"],
        icd10_codes=["Z96.89"],
        tier=ClinicalTier.TIER3_IMPORTANT,
        description="Support devices present — verify appropriate positioning",
    ),

    # === L. Trauma (1) ===
    CXRClinicalCondition(
        id=22, name="Rib / Skeletal Fracture",
        category="L. Trauma",
        cxr_patterns=["Fracture"],
        icd10_codes=["S22.30", "S22.40"],
        tier=ClinicalTier.TIER2_URGENT,
        description="Osseous fracture identified — correlate with trauma mechanism",
    ),

    # === M. No Acute Finding (1) ===
    CXRClinicalCondition(
        id=23, name="No Acute Cardiopulmonary Abnormality",
        category="M. Normal",
        cxr_patterns=["No Finding"],
        icd10_codes=["Z03.89"],
        tier=ClinicalTier.TIER3_IMPORTANT,
        description="No significant pathology detected on CXR",
    ),
]

CONDITION_BY_ID = {c.id: c for c in CXR_CLINICAL_CONDITIONS}

TORCHXRAYVISION_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices",
    "Infiltration", "Emphysema", "Fibrosis", "Hernia",
]


def map_cxr_to_clinical(
    pathology_probs: dict[str, float],
    threshold: float = 0.3,
) -> list[dict]:
    """
    Map CXR pathology probabilities to clinical diagnoses.

    Args:
        pathology_probs: dict of {pathology_name: probability}
        threshold: minimum probability to consider a pathology present

    Returns:
        List of clinical diagnoses sorted by tier (urgency) then confidence.
    """
    results = []

    for condition in CXR_CLINICAL_CONDITIONS:
        effective_threshold = max(threshold, condition.min_probability)

        present_patterns = []
        max_prob = 0.0
        for pattern in condition.cxr_patterns:
            prob = pathology_probs.get(pattern, 0.0)
            if prob >= effective_threshold:
                present_patterns.append(pattern)
                max_prob = max(max_prob, prob)

        triggered = False
        if condition.requires_all:
            triggered = len(present_patterns) == len(condition.cxr_patterns)
        else:
            triggered = len(present_patterns) > 0

        if not triggered:
            continue

        if condition.requires_all:
            confidence = sum(
                pathology_probs.get(p, 0) for p in condition.cxr_patterns
            ) / len(condition.cxr_patterns)
        else:
            confidence = max_prob

        results.append({
            "id": condition.id,
            "name": condition.name,
            "category": condition.category,
            "tier": condition.tier.value,
            "tier_label": condition.tier.name.replace("_", " ").title(),
            "confidence": round(confidence, 3),
            "description": condition.description,
            "icd10_codes": condition.icd10_codes,
            "triggering_pathologies": present_patterns,
        })

    results.sort(key=lambda r: (r["tier"], -r["confidence"]))
    return results
