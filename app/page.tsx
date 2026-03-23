"use client";

import { useState, useCallback } from "react";
import type { CXRAnalysisResult, ReportResult } from "@/types/cxr";
import UploadZone from "@/components/UploadZone";
import FindingsPanel from "@/components/FindingsPanel";
import ClinicalDiagnosisPanel from "@/components/ClinicalDiagnosisPanel";
import ReportPanel from "@/components/ReportPanel";
import LoadingSpinner from "@/components/LoadingSpinner";

type AppState = "idle" | "analyzing" | "generating-report" | "done" | "error";

export default function Home() {
  const [state, setState] = useState<AppState>("idle");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<CXRAnalysisResult | null>(null);
  const [report, setReport] = useState<ReportResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = useCallback((file: File) => {
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target?.result as string);
    reader.readAsDataURL(file);
    setAnalysis(null);
    setReport(null);
    setError(null);
    setState("idle");
  }, []);

  // Direct Railway URL bypasses Vercel 10s timeout limit on Hobby plan.
  // Set NEXT_PUBLIC_CXR_BACKEND_URL in Vercel env vars to Railway service URL.
  const BACKEND = process.env.NEXT_PUBLIC_CXR_BACKEND_URL || "";
  const analyzeUrl = BACKEND ? `${BACKEND}/analyze` : "/api/analyze-cxr";
  const reportUrl = BACKEND ? `${BACKEND}/report` : "/api/report";

  const handleAnalyze = useCallback(async () => {
    if (!imageFile) return;
    setState("analyzing");
    setError(null);

    try {
      // Step 1: CXR classification
      const formData = new FormData();
      formData.append("cxr_image", imageFile);

      const analysisRes = await fetch(analyzeUrl, {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(300_000), // 5 min — Railway CPU cold start
      });
      const analysisJson = await analysisRes.json();

      if (!analysisRes.ok || !analysisJson.success) {
        throw new Error(analysisJson.error || `Analysis failed: ${analysisRes.status}`);
      }

      const analysisData: CXRAnalysisResult = analysisJson.data;
      setAnalysis(analysisData);

      // Step 2: Generate radiology report
      setState("generating-report");

      try {
        const reportRes = await fetch(reportUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            findings: analysisData.findings,
            urgency_level: analysisData.urgency_level,
            confidence_score: analysisData.confidence_score,
            no_finding_probability: analysisData.no_finding_probability,
            clinical_diagnoses: analysisData.clinical_diagnoses || [],
          }),
          signal: AbortSignal.timeout(300_000), // 5 min — Claude CLI can take 90s
        });
        const reportJson = await reportRes.json();
        // Show report if available; gracefully skip if generation failed
        if (reportRes.ok && reportJson.success && reportJson.data) {
          setReport(reportJson.data);
        }
        // Analysis findings are already shown — don't block on report failure
      } catch {
        // Report generation failed; findings panel still visible
      }

      setState("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error occurred");
      setState("error");
    }
  }, [imageFile]);

  const handleReset = () => {
    setImageFile(null);
    setImagePreview(null);
    setAnalysis(null);
    setReport(null);
    setError(null);
    setState("idle");
  };

  const isLoading = state === "analyzing" || state === "generating-report";

  const URGENCY_HEADER_COLORS: Record<string, string> = {
    critical: "bg-red-600",
    urgent: "bg-orange-500",
    routine: "bg-yellow-500",
    normal: "bg-green-500",
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">CXR Analyzer</h1>
              <p className="text-xs text-gray-500">AI-Powered Chest X-Ray Interpretation</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {(analysis || imageFile) && (
              <button
                onClick={handleReset}
                className="flex items-center gap-1.5 text-sm text-blue-600 hover:text-blue-700 font-medium transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M12 4v16m8-8H4"
                  />
                </svg>
                New Analysis
              </button>
            )}
            <div className="text-right text-xs text-gray-400 hidden sm:block">
              <p>TorchXRayVision DenseNet + CLIP Zero-Shot</p>
              <p>18-pathology + zero-shot classification</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left column: Upload + Image — on mobile after analysis, pushed below findings via order */}
          <div className={`space-y-4 ${analysis && !isLoading ? "order-2 lg:order-1" : "order-1"}`}>
            {/* Upload zone — always visible */}
            {(state === "idle" || state === "error" || !imagePreview) && (
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
                <h2 className="text-sm font-semibold text-gray-700 mb-3">Upload Chest X-Ray</h2>
                <UploadZone onFileSelect={handleFileSelect} disabled={isLoading} />
              </div>
            )}

            {/* Image Preview — before analysis: show here (mobile sees upload → image → button) */}
            {imagePreview && !analysis && (
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
                <h2 className="text-sm font-semibold text-gray-700 mb-3">Preview</h2>
                <img
                  src={imagePreview}
                  alt="Chest X-ray preview"
                  className="w-full rounded-lg object-contain max-h-80 bg-black"
                />
                {(state === "idle" || state === "error") && (
                  <button
                    onClick={handleAnalyze}
                    disabled={!imageFile}
                    className="mt-3 w-full py-2.5 bg-blue-600 text-white text-sm font-semibold rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    Analyze X-Ray
                  </button>
                )}
              </div>
            )}

            {/* Loading indicator — show in left column when analyzing */}
            {isLoading && (
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                <LoadingSpinner
                  message={
                    state === "analyzing"
                      ? "Analyzing chest X-ray with DenseNet + CLIP Zero-Shot..."
                      : "Generating radiology report with Claude..."
                  }
                />
              </div>
            )}

            {/* After analysis: Image + GradCAM moves below findings on mobile (order-last on mobile) */}
            {imagePreview && analysis && (
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4 order-last lg:order-none">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-sm font-semibold text-gray-700">Image & GradCAM</h2>
                  <span
                    className={`text-xs font-bold text-white px-2 py-1 rounded ${
                      URGENCY_HEADER_COLORS[analysis.urgency_level] || "bg-gray-500"
                    }`}
                  >
                    {analysis.urgency_level.toUpperCase()}
                  </span>
                </div>
                {/* Original preview — always visible */}
                <div className="space-y-1 mb-2">
                  <p className="text-xs text-gray-400 font-medium">Original</p>
                  <img
                    src={imagePreview}
                    alt="Chest X-ray preview"
                    className="w-full rounded-lg object-contain max-h-64 bg-black"
                  />
                </div>
                {/* GradCAM overlay — shown when available */}
                {analysis.heatmap_base64 && (
                  <div className="space-y-1">
                    <p className="text-xs text-gray-400 font-medium">AI Attention Map (GradCAM)</p>
                    <img
                      src={`data:image/png;base64,${analysis.heatmap_base64}`}
                      alt="CXR with GradCAM heatmap"
                      className="w-full rounded-lg object-contain max-h-64"
                    />
                    <p className="text-xs text-gray-400 text-center">
                      highlighted region drove AI decision
                    </p>
                  </div>
                )}
                <button
                  onClick={handleReset}
                  className="mt-3 w-full py-2.5 bg-gray-100 text-gray-700 text-sm font-semibold rounded-lg hover:bg-gray-200 transition-colors"
                >
                  Upload New X-Ray
                </button>
              </div>
            )}

            {/* Upload another button when no preview */}
            {!imagePreview && (
              <button
                onClick={handleReset}
                className="w-full py-2.5 text-sm text-gray-500 hover:text-gray-700 underline"
              >
                Clear
              </button>
            )}
          </div>

          {/* Right column: Results — on mobile after analysis, shown FIRST via order */}
          <div className={`space-y-4 ${analysis && !isLoading ? "order-1 lg:order-2" : "order-2"}`}>
            {/* Error */}
            {state === "error" && error && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-4">
                <p className="text-sm font-semibold text-red-700">Analysis Failed</p>
                <p className="text-sm text-red-600 mt-1">{error}</p>
              </div>
            )}

            {/* Clinical Diagnoses — tier-based clinical interpretation */}
            {analysis && !isLoading && analysis.clinical_diagnoses?.length > 0 && (
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
                <ClinicalDiagnosisPanel diagnoses={analysis.clinical_diagnoses} />
              </div>
            )}

            {/* Findings — DenseNet AI pathology scores */}
            {analysis && !isLoading && (
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
                <FindingsPanel result={analysis} />
              </div>
            )}

            {/* Report */}
            {report && !isLoading && (
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
                <ReportPanel report={report} />
              </div>
            )}

            {/* Idle placeholder */}
            {state === "idle" && !imageFile && (
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-8 flex flex-col items-center justify-center gap-3 text-center min-h-48">
                <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center">
                  <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-600">Results will appear here</p>
                  <p className="text-xs text-gray-400 mt-1">
                    Upload a CXR image to begin AI analysis
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-8 py-4 border-t border-gray-200 text-center">
        <p className="text-xs text-gray-400">
          For research and educational use only. Not for clinical decision-making without physician oversight.
        </p>
      </footer>
    </div>
  );
}
