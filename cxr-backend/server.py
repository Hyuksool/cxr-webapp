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

    Tries claude_agent_sdk first (Claude CLI), then falls back to
    the anthropic Python SDK when the CLI subprocess fails or isn't available.
    Both approaches require ANTHROPIC_API_KEY.
    """
    # --- Try claude_agent_sdk (Claude Code CLI) ---
    try:
        from claude_agent_sdk import query as claude_query, ClaudeAgentOptions

        def _extract_text(value) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return "".join(_extract_text(item) for item in value)
            if isinstance(value, dict):
                return _extract_text(value.get("text", ""))
            if hasattr(value, "text"):
                return _extract_text(value.text)
            return str(value) if value else ""

        result = ""
        async for msg in claude_query(prompt=prompt, options=ClaudeAgentOptions(model="sonnet")):
            if hasattr(msg, "content"):
                result += _extract_text(msg.content)
        if result.strip():
            return result
        raise RuntimeError("claude_agent_sdk returned empty response")
    except Exception as sdk_err:
        print(f"claude_agent_sdk failed ({sdk_err}), falling back to anthropic SDK")

    # --- Fallback: anthropic Python SDK ---
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set and claude_agent_sdk also failed")

    client = anthropic.Anthropic(api_key=api_key)
    message = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        ),
    )
    return message.content[0].text


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print("Starting CXR Analysis Server...")
    success = preload_cxr_model()
    if success:
        print("CXR model loaded successfully")
    else:
        print("WARNING: CXR model not loaded — classification disabled")

    zs_success = preload_zero_shot_model()
    if zs_success:
        print("Zero-shot CLIP model loaded successfully")
    else:
        print("INFO: Zero-shot model not loaded — zero-shot analysis disabled")


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
        result = classify_cxr_image(image_bytes)

        # Run zero-shot analysis (independent of TorchXRayVision)
        zs_result = classify_zero_shot(image_bytes)

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
        return ReportResponse(
            success=False,
            error=f"Report generation failed: {type(e).__name__}: {e}",
        )
