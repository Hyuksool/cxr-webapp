"use client";

import type { CXRAnalysisResult } from "@/types/cxr";

interface FindingsPanelProps {
  result: CXRAnalysisResult;
}

const URGENCY_COLORS = {
  critical: { bg: "bg-red-50", border: "border-red-200", badge: "bg-red-100 text-red-700", bar: "bg-red-500" },
  urgent: { bg: "bg-orange-50", border: "border-orange-200", badge: "bg-orange-100 text-orange-700", bar: "bg-orange-500" },
  routine: { bg: "bg-yellow-50", border: "border-yellow-200", badge: "bg-yellow-100 text-yellow-700", bar: "bg-yellow-500" },
  info: { bg: "bg-blue-50", border: "border-blue-200", badge: "bg-blue-100 text-blue-700", bar: "bg-blue-500" },
  normal: { bg: "bg-green-50", border: "border-green-200", badge: "bg-green-100 text-green-700", bar: "bg-green-500" },
};

const URGENCY_LABELS = {
  critical: "CRITICAL",
  urgent: "URGENT",
  routine: "ROUTINE",
  normal: "NORMAL",
  info: "INFO",
};

// Client-side primary diagnosis synthesis from findings
const DIAGNOSIS_GROUPS: Record<string, string[]> = {
  "Pneumonia / Infection": ["Pneumonia", "Consolidation", "Infiltration"],
  "Pulmonary Edema / CHF": ["Edema", "Cardiomegaly", "Pleural Effusion", "Effusion"],
  "Pneumothorax": ["Pneumothorax"],
  "Mass / Nodule": ["Mass", "Nodule", "Lung Lesion"],
  "Interstitial Disease": ["Fibrosis", "Emphysema"],
};

function derivePrimaryDiagnosis(findings: CXRAnalysisResult["findings"]): { primary: string; differentials: string[] } {
  if (findings.length === 0) return { primary: "No significant pathology", differentials: [] };

  const groupScores: Record<string, number> = {};
  for (const [group, members] of Object.entries(DIAGNOSIS_GROUPS)) {
    let score = 0;
    for (const f of findings) {
      if (members.includes(f.name)) score += f.probability;
    }
    if (score > 0) groupScores[group] = score;
  }

  if (Object.keys(groupScores).length === 0) {
    return { primary: findings[0].name, differentials: findings.slice(1, 3).map(f => f.name) };
  }

  const sorted = Object.entries(groupScores).sort((a, b) => b[1] - a[1]);
  const primary = sorted[0][0];
  const differentials = sorted.slice(1, 3).filter(([, v]) => v > 0.3).map(([k]) => k);

  return { primary, differentials };
}

export default function FindingsPanel({ result }: FindingsPanelProps) {
  const urgencyColors = URGENCY_COLORS[result.urgency_level] || URGENCY_COLORS.normal;
  const zs = result.zero_shot;
  const { primary, differentials } = derivePrimaryDiagnosis(result.findings);

  // Split findings: significant (≥20%) vs minor
  const significantFindings = result.findings.filter(f => f.probability >= 0.2);
  const minorFindings = result.findings.filter(f => f.probability < 0.2);

  return (
    <div className="space-y-4">
      {/* PRIMARY DIAGNOSIS — most prominent element */}
      {result.findings.length > 0 && (
        <div className={`rounded-lg p-4 border-2 ${
          result.urgency_level === "critical" ? "border-red-500 bg-red-50" :
          result.urgency_level === "urgent" ? "border-orange-500 bg-orange-50" :
          "border-blue-500 bg-blue-50"
        }`}>
          <p className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-1">Primary Diagnosis</p>
          <p className={`text-xl font-bold ${
            result.urgency_level === "critical" ? "text-red-800" :
            result.urgency_level === "urgent" ? "text-orange-800" :
            "text-blue-800"
          }`}>
            {primary}
          </p>
          {differentials.length > 0 && (
            <p className="text-sm text-gray-600 mt-1">
              Differential: {differentials.join(", ")}
            </p>
          )}
          <div className="flex items-center gap-3 mt-2">
            <span className={`text-xs font-bold px-2 py-1 rounded ${urgencyColors.badge}`}>
              {URGENCY_LABELS[result.urgency_level]}
            </span>
            <span className="text-sm font-bold text-gray-700">
              {(result.confidence_score * 100).toFixed(0)}% confidence
            </span>
          </div>
        </div>
      )}

      {/* No findings banner */}
      {result.findings.length === 0 && (
        <div className="rounded-lg p-4 border bg-green-50 border-green-200">
          <p className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-1">Diagnosis</p>
          <p className="text-xl font-bold text-green-800">No significant pathology detected</p>
          <span className="text-xs font-bold px-2 py-1 rounded bg-green-100 text-green-700 mt-2 inline-block">
            NORMAL
          </span>
        </div>
      )}

      {/* Significant Findings (≥20% probability) */}
      {significantFindings.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">
            Key Findings — DenseNet AI Analysis
          </h3>
          <div className="space-y-2">
            {significantFindings.map((finding) => {
              const fc = URGENCY_COLORS[finding.urgency] || URGENCY_COLORS.routine;
              return (
                <div key={finding.name} className="flex items-center gap-3 p-3 rounded-lg bg-white border border-gray-100 shadow-sm">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-gray-800">{finding.name}</span>
                      <span className={`text-xs px-1.5 py-0.5 rounded ${fc.badge}`}>
                        {URGENCY_LABELS[finding.urgency]}
                      </span>
                    </div>
                    <div className="mt-1 h-1.5 bg-gray-100 rounded-full">
                      <div
                        className={`h-full rounded-full ${fc.bar}`}
                        style={{ width: `${finding.probability * 100}%` }}
                      />
                    </div>
                  </div>
                  <span className="text-sm font-bold text-gray-700 w-12 text-right">
                    {(finding.probability * 100).toFixed(0)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Minor Findings (collapsed) */}
      {minorFindings.length > 0 && (
        <details className="group">
          <summary className="cursor-pointer text-xs text-gray-400 hover:text-gray-600 select-none">
            {minorFindings.length} additional findings below 20% ▶
          </summary>
          <div className="mt-2 space-y-1">
            {minorFindings.map((finding) => (
              <div key={finding.name} className="flex items-center gap-2 py-1">
                <span className="text-xs text-gray-600 w-44 truncate">{finding.name}</span>
                <div className="flex-1 h-1 bg-gray-100 rounded-full">
                  <div
                    className="h-full bg-gray-400 rounded-full"
                    style={{ width: `${finding.probability * 100}%` }}
                  />
                </div>
                <span className="text-xs text-gray-500 w-10 text-right">
                  {(finding.probability * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Zero-Shot CLIP Findings */}
      {zs && zs.available && (
        <div>
          <div className="flex items-center gap-2 mb-2">
            <h3 className="text-sm font-semibold text-gray-700">
              Zero-Shot Analysis — CLIP
            </h3>
            <span className="text-xs bg-purple-100 text-purple-700 px-1.5 py-0.5 rounded">
              CheXzero-style
            </span>
          </div>

          {zs.findings.length === 0 ? (
            <p className="text-xs text-gray-500 italic">No pathologies detected above threshold by zero-shot model</p>
          ) : (
            <div className="space-y-2">
              {zs.findings.map((finding) => {
                const fc = URGENCY_COLORS[finding.urgency as keyof typeof URGENCY_COLORS] || URGENCY_COLORS.routine;
                return (
                  <div key={finding.name} className="flex items-center gap-3 p-3 rounded-lg bg-purple-50 border border-purple-100">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-gray-800">{finding.name}</span>
                        <span className={`text-xs px-1.5 py-0.5 rounded ${fc.badge}`}>
                          {URGENCY_LABELS[finding.urgency as keyof typeof URGENCY_LABELS] || finding.urgency}
                        </span>
                      </div>
                      <div className="mt-1 h-1.5 bg-purple-100 rounded-full">
                        <div
                          className="h-full bg-purple-500 rounded-full"
                          style={{ width: `${finding.similarity * 100}%` }}
                        />
                      </div>
                    </div>
                    <span className="text-sm font-bold text-purple-700 w-12 text-right">
                      {(finding.similarity * 100).toFixed(0)}%
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {/* All zero-shot scores */}
          <details className="mt-2 group">
            <summary className="cursor-pointer text-xs text-gray-400 hover:text-gray-600 select-none">
              Show all zero-shot similarity scores ▶
            </summary>
            <div className="mt-2 space-y-1 max-h-48 overflow-y-auto">
              {zs.pathologies.map((p) => (
                <div key={p.name} className="flex items-center gap-2 py-0.5">
                  <span className="text-xs text-gray-600 w-44 truncate">{p.name}</span>
                  <div className="flex-1 h-1 bg-gray-100 rounded-full">
                    <div
                      className="h-full bg-purple-400 rounded-full"
                      style={{ width: `${p.similarity * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-500 w-10 text-right">
                    {(p.similarity * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </details>
          <p className="text-xs text-gray-400 mt-1">{zs.model}</p>
        </div>
      )}

      {/* Zero-shot unavailable notice */}
      {zs && !zs.available && (
        <div className="text-xs text-gray-400 italic">
          Zero-shot analysis not available (CLIP model not loaded)
        </div>
      )}

      {/* No Finding */}
      {result.no_finding_probability > 0 && (
        <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50 border border-gray-200">
          <span className="text-sm text-gray-600">No Finding probability</span>
          <span className="text-sm font-bold text-gray-700">
            {(result.no_finding_probability * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {/* All Pathologies */}
      <details className="group">
        <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700 select-none">
          Show all {result.pathologies.length} DenseNet pathology scores ▶
        </summary>
        <div className="mt-2 space-y-1 max-h-64 overflow-y-auto">
          {result.pathologies.map((p) => (
            <div key={p.name} className="flex items-center gap-2 py-1">
              <span className="text-xs text-gray-600 w-48 truncate">{p.name}</span>
              <div className="flex-1 h-1 bg-gray-100 rounded-full">
                <div
                  className="h-full bg-blue-400 rounded-full"
                  style={{ width: `${p.probability * 100}%` }}
                />
              </div>
              <span className="text-xs text-gray-500 w-10 text-right">
                {(p.probability * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}
