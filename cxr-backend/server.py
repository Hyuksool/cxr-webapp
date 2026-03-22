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


class CXRAnalysisResponse(BaseModel):
    success: bool
    data: CXRClassificationResult | None = None
    error: str | None = None


class ReportRequest(BaseModel):
    findings: list[dict]
    urgency_level: str
    confidence_score: float
    no_finding_probability: float


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

        return CXRAnalysisResponse(
            success=True,
            data=CXRClassificationResult(
                **result,
                zero_shot=ZeroShotResult(**zs_result),
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


@app.post("/report", response_model=ReportResponse)
async def generate_report(req: ReportRequest):
    """
    Generate radiology report using Claude based on CXR findings.
    """
    try:
        # Build findings summary for Claude
        if not req.findings:
            findings_text = "No significant findings detected."
        else:
            findings_lines = []
            for f in req.findings:
                findings_lines.append(
                    f"- {f['name']}: {f['probability']*100:.1f}% probability ({f['urgency']} urgency)"
                )
            findings_text = "\n".join(findings_lines)

        prompt = f"""You are an expert radiologist. Generate a structured chest X-ray radiology report based on AI analysis results.

AI Analysis Results:
Urgency Level: {req.urgency_level.upper()}
Overall Confidence: {req.confidence_score*100:.1f}%
No Finding Probability: {req.no_finding_probability*100:.1f}%

Detected Findings:
{findings_text}

Generate a radiology report with these sections:
1. CLINICAL INDICATION: Brief indication for CXR
2. TECHNIQUE: Standard PA/AP chest radiograph
3. FINDINGS: Describe each finding professionally
4. IMPRESSION: Concise summary (most important 1-3 points)
5. RECOMMENDATIONS: Clinical action items based on urgency

Important:
- Write as a professional radiologist
- Use appropriate medical terminology
- Flag critical/urgent findings prominently
- Be concise but complete
- If no findings, state clearly normal study
- This is AI-assisted, not a replacement for clinical judgment"""

        report_text = await _call_claude(prompt)

        # Parse sections
        sections = {}
        current_section = None
        current_lines = []

        for line in report_text.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("1.", "2.", "3.", "4.", "5.")):
                if current_section and current_lines:
                    sections[current_section] = "\n".join(current_lines).strip()
                # Extract section name
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    current_section = parts[0].split(".", 1)[1].strip()
                    first_content = parts[1].strip()
                    current_lines = [first_content] if first_content else []
                else:
                    current_section = stripped
                    current_lines = []
            elif current_section is not None:
                current_lines.append(line)

        if current_section and current_lines:
            sections[current_section] = "\n".join(current_lines).strip()

        return ReportResponse(
            success=True,
            data={
                "report": report_text,
                "sections": sections,
                "urgency_level": req.urgency_level,
                "model": "claude-agent-sdk/sonnet",
            },
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Always return success=True with fallback report so UI can display findings.
        # Never return success=False from this endpoint.
        _err_str = f"{type(e).__name__}: {e}"
        try:
            findings_text = "\n".join(
                f"  - {f.get('name', 'Unknown')}: {f.get('probability', 0)*100:.1f}% probability ({f.get('urgency', 'unknown')} urgency)"
                for f in (req.findings or [])
            ) or "  No significant findings detected."
            findings_summary = "\n".join(
                f"{f.get('name', 'Unknown')}: {f.get('probability', 0)*100:.1f}% ({f.get('urgency', 'unknown')})"
                for f in (req.findings or [])
            ) or "No significant findings."
        except Exception:
            findings_text = "  Unable to parse findings."
            findings_summary = "Unable to parse findings."

        fallback_text = "\n".join([
            "CLINICAL INDICATION: AI-assisted chest X-ray analysis.",
            "",
            "TECHNIQUE: Standard chest radiograph (AI analysis only).",
            "",
            "FINDINGS:",
            findings_text,
            "",
            f"IMPRESSION: {req.urgency_level.upper()} urgency. Confidence: {req.confidence_score*100:.1f}%.",
            "",
            "RECOMMENDATIONS: Clinical correlation required. AI narrative report unavailable — consult a radiologist.",
            "",
            f"NOTE: Automated report generation failed ({_err_str}). "
            "This is a fallback summary based on raw AI findings only.",
        ])
        return ReportResponse(
            success=True,
            data={
                "report": fallback_text,
                "sections": {
                    "CLINICAL INDICATION": "AI-assisted chest X-ray analysis.",
                    "FINDINGS": findings_summary,
                    "IMPRESSION": f"{req.urgency_level.upper()} urgency. Confidence: {req.confidence_score*100:.1f}%.",
                    "RECOMMENDATIONS": "Clinical correlation required. Consult a radiologist.",
                },
                "urgency_level": req.urgency_level,
                "model": "fallback (Claude CLI unavailable)",
                "error": _err_str,
            },
        )
