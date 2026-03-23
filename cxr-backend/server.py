"""
CXR Analysis FastAPI Server.

Uses TorchXRayVision DenseNet for 18-pathology classification,
GradCAM for localization, and Claude CLI/SDK for radiology report generation.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8200
"""

import os
import sys
from pathlib import Path

import asyncio
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from cxr_classifier import classify_cxr_image, preload_cxr_model
from cxr_zero_shot import classify_zero_shot, preload_zero_shot_model
from cxr_clinical_mapper import map_cxr_to_clinical


app = FastAPI(
    title="CXR Analysis Server",
    description="Chest X-Ray AI analysis using TorchXRayVision + Claude",
    version="1.0.0",
)

_ALLOWED_ORIGINS = [
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_FRONTEND_URL = os.environ.get("FRONTEND_URL", "")
if _FRONTEND_URL:
    _ALLOWED_ORIGINS.append(_FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Claude CLI/SDK for report generation (no direct Anthropic API calls)


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class PathologyResult(BaseModel):
    name: str
    probability: float
    urgency: str


class ZeroShotPathology(BaseModel):
    name: str
    similarity: float
    urgency: str


class ZeroShotResult(BaseModel):
    available: bool
    pathologies: list[ZeroShotPathology]
    findings: list[ZeroShotPathology]
    model: str


class CXRClassificationResult(BaseModel):
    pathologies: list[PathologyResult]
    findings: list[PathologyResult]
    urgency_level: str
    heatmap_base64: str | None = None
    model_pathologies: list[str]
    confidence_score: float
    no_finding_probability: float
    zero_shot: ZeroShotResult | None = None
    clinical_diagnoses: list[dict] = []


class CXRAnalysisResponse(BaseModel):
    success: bool
    data: CXRClassificationResult | None = None
    error: str | None = None


class ReportRequest(BaseModel):
    findings: list[dict]
    urgency_level: str
    confidence_score: float
    no_finding_probability: float
    clinical_diagnoses: list[dict] = []


class ReportResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None


# ─────────────────────────────────────────────
# Claude helpers
# ─────────────────────────────────────────────

async def _call_claude(prompt: str) -> str:
    """Call Claude for text generation.

    Strategy:
    1. If ANTHROPIC_API_KEY is set (Railway/deployed): use Anthropic SDK directly
    2. Otherwise (local dev with Claude Max): use claude_agent_sdk CLI
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if api_key:
        # Direct Anthropic SDK — works in Railway Docker without CLI auth
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        result_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                result_text += block.text
        if not result_text.strip():
            raise RuntimeError("Anthropic API returned empty response")
        return result_text

    # Fallback: Claude CLI/SDK (local dev with Claude Max subscription)
    from claude_agent_sdk import query as claude_query, ClaudeAgentOptions
    result_text = ""
    async for msg in claude_query(prompt=prompt, options=ClaudeAgentOptions(model="sonnet")):
        if hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        result_text += block.text
                    elif isinstance(block, dict) and "text" in block:
                        result_text += block["text"]
            elif isinstance(content, str):
                result_text += content

    if not result_text.strip():
        raise RuntimeError("Claude CLI returned empty response")
    return result_text


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print("Starting CXR Analysis Server...")
    loop = asyncio.get_event_loop()

    # Load both models in parallel threads to reduce cold-start time
    densenet_ok, clip_ok = await asyncio.gather(
        loop.run_in_executor(None, preload_cxr_model),
        loop.run_in_executor(None, preload_zero_shot_model),
    )

    print("CXR model loaded successfully" if densenet_ok else "WARNING: CXR model not loaded — classification disabled")
    print("Zero-shot CLIP model loaded successfully" if clip_ok else "INFO: Zero-shot model not loaded — zero-shot analysis disabled")


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    from cxr_zero_shot import _CLIP_MODEL
    return {
        "status": "ok",
        "model": "TorchXRayVision DenseNet densenet121-res224-all",
        "zero_shot": "CLIP openai/clip-vit-base-patch32" if _CLIP_MODEL is not None else "disabled",
        "version": "1.1.0",
    }


@app.post("/analyze", response_model=CXRAnalysisResponse)
async def analyze_cxr(cxr_image: UploadFile = File(...)):
    """
    Analyze chest X-ray image.

    Returns 18-pathology classification probabilities, urgency level,
    and optional GradCAM heatmap.
    """
    # Validate file type
    if cxr_image.content_type and not cxr_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await cxr_image.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(image_bytes) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=400, detail="File too large (max 20MB)")

    try:
        loop = asyncio.get_event_loop()

        # Run both models in parallel threads (non-blocking async event loop)
        result, zs_result = await asyncio.gather(
            loop.run_in_executor(None, classify_cxr_image, image_bytes),
            loop.run_in_executor(None, classify_zero_shot, image_bytes),
        )

        # Map pathology probabilities to clinical diagnoses
        pathology_probs = {p["name"]: p["probability"] for p in result["pathologies"]}
        clinical_diagnoses = map_cxr_to_clinical(pathology_probs)

        return CXRAnalysisResponse(
            success=True,
            data=CXRClassificationResult(
                **result,
                zero_shot=ZeroShotResult(**zs_result),
                clinical_diagnoses=clinical_diagnoses,
            ),
        )
    except RuntimeError as e:
        return CXRAnalysisResponse(success=False, error=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return CXRAnalysisResponse(
            success=False,
            error=f"Analysis failed: {type(e).__name__}: {e}",
        )


# ─────────────────────────────────────────────
# Clinical interpretation rules (no LLM dependency)
# ─────────────────────────────────────────────

# Maps pathology findings to clinical diagnosis descriptions
CLINICAL_DESCRIPTIONS: dict[str, str] = {
    "Atelectasis": "Partial collapse of lung parenchyma, likely subsegmental atelectasis. May represent post-obstructive changes or hypoventilation.",
    "Cardiomegaly": "Cardiothoracic ratio exceeds normal limits, suggesting cardiomegaly. Consider echocardiography for further evaluation.",
    "Consolidation": "Airspace consolidation identified, consistent with pneumonia, hemorrhage, or organizing process. Clinical correlation with symptoms and labs recommended.",
    "Edema": "Pulmonary edema pattern identified. Findings may represent cardiogenic (CHF) or non-cardiogenic (ARDS) etiology.",
    "Enlarged Cardiomediastinum": "Widened mediastinal silhouette. Consider aortic pathology, lymphadenopathy, or mass lesion. CT recommended if clinically indicated.",
    "Fracture": "Osseous abnormality suggesting fracture. Clinical correlation with trauma history recommended.",
    "Lung Lesion": "Focal parenchymal lesion identified. Differential includes neoplasm, granuloma, or infectious focus. Follow-up imaging or CT recommended.",
    "Lung Opacity": "Pulmonary opacity identified. Differential includes infectious process, atelectasis, or mass. Correlate with clinical presentation.",
    "Pleural Effusion": "Pleural fluid collection identified. Consider infectious, malignant, or cardiac etiology.",
    "Pleural Other": "Pleural abnormality identified. May represent thickening, calcification, or other pleural pathology.",
    "Pneumonia": "Findings consistent with pneumonia. Recommend clinical correlation with symptoms, labs (CBC, CRP), and consideration of antibiotic therapy.",
    "Pneumothorax": "Pneumothorax identified. Assess clinical stability and consider chest tube placement if symptomatic or tension physiology suspected.",
    "Support Devices": "Medical support devices identified (e.g., central line, ETT, NG tube). Verify appropriate positioning.",
    "Infiltration": "Pulmonary infiltrate identified. Differential includes infection, inflammation, or hemorrhage.",
    "Emphysema": "Hyperinflated lungs with findings suggestive of emphysema/COPD.",
    "Fibrosis": "Interstitial changes suggestive of pulmonary fibrosis. Consider HRCT for further characterization.",
    "Hernia": "Diaphragmatic hernia suspected. Consider CT for confirmation and surgical consultation.",
    "Mass": "Pulmonary mass lesion identified. Urgent CT and potential tissue sampling recommended to evaluate for malignancy.",
    "Nodule": "Pulmonary nodule identified. Follow Fleischner Society guidelines for management based on size and risk factors.",
    "Effusion": "Pleural effusion identified. Consider thoracentesis if clinically indicated.",
}

URGENCY_RECOMMENDATIONS: dict[str, str] = {
    "critical": "URGENT: Immediate clinical assessment required. Consider emergent intervention.",
    "urgent": "Prompt clinical evaluation recommended. Correlate with patient symptoms and consider further imaging.",
    "routine": "Routine follow-up recommended. Clinical correlation advised.",
    "normal": "No acute findings requiring immediate intervention.",
    "info": "Incidental finding noted. Clinical correlation as needed.",
}

# Pathology groupings for synthesizing primary diagnosis
DIAGNOSIS_GROUPS: dict[str, list[str]] = {
    "Pneumonia / Infection": ["Pneumonia", "Consolidation", "Infiltration"],
    "Pulmonary Edema / CHF": ["Edema", "Cardiomegaly", "Pleural Effusion", "Effusion"],
    "Pneumothorax": ["Pneumothorax"],
    "Mass / Nodule": ["Mass", "Nodule", "Lung Lesion"],
    "Interstitial Disease": ["Fibrosis", "Emphysema"],
    "Structural": ["Atelectasis", "Enlarged Cardiomediastinum", "Hernia"],
}


def _synthesize_primary_diagnosis(findings: list[dict]) -> str:
    """Derive primary clinical diagnosis from pathology findings."""
    if not findings:
        return "No significant pathology detected. Normal study."

    # Score each diagnostic group
    group_scores: dict[str, float] = {}
    for group_name, members in DIAGNOSIS_GROUPS.items():
        score = 0.0
        for f in findings:
            if f.get("name") in members:
                score += f.get("probability", 0)
        if score > 0:
            group_scores[group_name] = score

    if not group_scores:
        top = findings[0]
        return f"{top['name']} (probability: {top['probability']*100:.0f}%)"

    # Primary = highest aggregate score
    primary = max(group_scores, key=group_scores.get)  # type: ignore[arg-type]
    primary_prob = group_scores[primary]

    # Secondary diagnoses
    secondaries = sorted(
        [(k, v) for k, v in group_scores.items() if k != primary and v > 0.3],
        key=lambda x: x[1],
        reverse=True,
    )

    result = f"Primary: {primary}"
    if secondaries:
        sec_names = [s[0] for s in secondaries[:2]]
        result += f"\nDifferential: {', '.join(sec_names)}"

    return result


def _generate_rule_based_report(req: ReportRequest) -> dict:
    """Generate structured radiology report using clinical rules (no LLM)."""
    findings = req.findings or []

    # Sort by probability descending
    sorted_findings = sorted(findings, key=lambda f: f.get("probability", 0), reverse=True)

    # Primary diagnosis synthesis
    primary_dx = _synthesize_primary_diagnosis(sorted_findings)

    # Build FINDINGS section
    findings_lines = []
    for f in sorted_findings:
        name = f.get("name", "Unknown")
        prob = f.get("probability", 0)
        desc = CLINICAL_DESCRIPTIONS.get(name, f"{name} identified.")
        if prob >= 0.5:
            findings_lines.append(f"- {name} ({prob*100:.0f}%): {desc}")
        elif prob >= 0.2:
            findings_lines.append(f"- {name} ({prob*100:.0f}%): Possible {name.lower()}. {desc}")

    if not findings_lines:
        findings_text = "No significant abnormalities identified."
    else:
        findings_text = "\n".join(findings_lines)

    # Build IMPRESSION
    critical_findings = [f for f in sorted_findings if f.get("urgency") == "critical" and f.get("probability", 0) >= 0.3]
    urgent_findings = [f for f in sorted_findings if f.get("urgency") == "urgent" and f.get("probability", 0) >= 0.3]

    impression_parts = []
    if critical_findings:
        names = ", ".join(f.get("name", "") for f in critical_findings)
        impression_parts.append(f"CRITICAL: {names} — immediate evaluation required.")
    if urgent_findings:
        names = ", ".join(f.get("name", "") for f in urgent_findings)
        impression_parts.append(f"URGENT: {names} — prompt clinical correlation recommended.")

    if not impression_parts:
        top3 = sorted_findings[:3]
        if top3:
            names = ", ".join(f"{f.get('name', '')} ({f.get('probability', 0)*100:.0f}%)" for f in top3)
            impression_parts.append(f"Top findings: {names}. Clinical correlation advised.")
        else:
            impression_parts.append("No significant abnormalities.")

    impression_text = " ".join(impression_parts)

    # Recommendations
    rec = URGENCY_RECOMMENDATIONS.get(req.urgency_level, URGENCY_RECOMMENDATIONS["routine"])

    sections = {
        "PRIMARY DIAGNOSIS": primary_dx,
        "CLINICAL INDICATION": "AI-assisted chest X-ray screening and differential diagnosis.",
        "TECHNIQUE": "Standard PA/AP chest radiograph analyzed by TorchXRayVision DenseNet (18-pathology model) with CLIP zero-shot classification.",
        "FINDINGS": findings_text,
        "IMPRESSION": impression_text,
        "RECOMMENDATIONS": rec,
    }

    # Add Clinical Diagnoses section if available
    if req.clinical_diagnoses:
        dx_lines = []
        for dx in req.clinical_diagnoses:
            tier = dx.get("tier_label", "")
            name = dx.get("name", "")
            conf = dx.get("confidence", 0)
            icd = ", ".join(dx.get("icd10_codes", []))
            desc = dx.get("description", "")
            dx_lines.append(f"- [{tier}] {name} ({conf*100:.0f}%) [ICD-10: {icd}]\n  {desc}")
        sections["CLINICAL DIAGNOSES"] = "\n".join(dx_lines)

    report_text = "\n\n".join(f"{k}:\n{v}" for k, v in sections.items())

    return {
        "report": report_text,
        "sections": sections,
        "urgency_level": req.urgency_level,
        "model": "rule-based-clinical-v1",
        "primary_diagnosis": primary_dx,
    }


@app.post("/report", response_model=ReportResponse)
async def generate_report(req: ReportRequest):
    """
    Generate radiology report using rule-based clinical interpretation.
    Falls back to Claude LLM if available for enhanced narrative.
    """
    # Always generate rule-based report first (instant, no dependencies)
    rule_report = _generate_rule_based_report(req)

    # Try Claude LLM for enhanced narrative (optional overlay)
    try:
        findings_text = "\n".join(
            f"- {f['name']}: {f['probability']*100:.1f}% ({f['urgency']})"
            for f in (req.findings or [])
        ) or "No findings."

        prompt = f"""You are an expert radiologist. Write a concise clinical impression (3-5 sentences) for this CXR:

Findings: {findings_text}
Urgency: {req.urgency_level}
Confidence: {req.confidence_score*100:.0f}%

Write ONLY the clinical impression — no headers, no sections. Be specific and actionable."""

        claude_text = await _call_claude(prompt)
        if claude_text and claude_text.strip():
            rule_report["sections"]["CLINICAL IMPRESSION (AI)"] = claude_text.strip()
            rule_report["model"] = "rule-based-clinical-v1 + claude-enhanced"
    except Exception:
        pass  # Rule-based report is sufficient

    return ReportResponse(success=True, data=rule_report)
