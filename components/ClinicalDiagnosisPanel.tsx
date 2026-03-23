"use client";

interface ClinicalDiagnosis {
  id: number;
  name: string;
  category: string;
  tier: number;
  tier_label: string;
  confidence: number;
  description: string;
  icd10_codes: string[];
  triggering_pathologies: string[];
}

interface ClinicalDiagnosisPanelProps {
  diagnoses: ClinicalDiagnosis[];
}

export default function ClinicalDiagnosisPanel({ diagnoses }: ClinicalDiagnosisPanelProps) {
  const filtered = (diagnoses || [])
    .filter((dx) => dx.confidence >= 0.9)
    .sort((a, b) => b.confidence - a.confidence);

  if (filtered.length === 0) {
    return (
      <div className="rounded-lg p-4 bg-green-50 border border-green-200">
        <p className="text-sm font-semibold text-green-800">No high-confidence diagnoses</p>
        <p className="text-xs text-green-600 mt-1">No conditions exceeded 90% confidence threshold</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">Clinical Diagnoses</h3>
        <span className="text-xs text-gray-400">{filtered.length} condition{filtered.length !== 1 ? "s" : ""} ≥90%</span>
      </div>

      {filtered.map((dx) => (
        <div key={dx.id} className="rounded-lg p-3 border border-gray-200 bg-white">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-gray-900">{dx.name}</p>
              <p className="text-xs text-gray-500 mt-0.5">{dx.category}</p>
              <p className="text-xs text-gray-600 mt-1">{dx.description}</p>
            </div>
            <span className="text-sm font-bold text-gray-700 shrink-0">
              {(dx.confidence * 100).toFixed(0)}%
            </span>
          </div>

          <div className="mt-2 flex flex-wrap gap-1">
            {dx.icd10_codes.map((code) => (
              <span key={code} className="text-xs bg-gray-50 text-gray-600 px-1.5 py-0.5 rounded border border-gray-200">
                {code}
              </span>
            ))}
          </div>

          {dx.triggering_pathologies.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {dx.triggering_pathologies.map((p) => (
                <span key={p} className="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded">
                  {p}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
