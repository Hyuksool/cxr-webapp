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

const TIER_STYLES: Record<number, { bg: string; border: string; badge: string; icon: string }> = {
  1: { bg: "bg-red-50", border: "border-red-300", badge: "bg-red-600 text-white", icon: "TIER 1 CRITICAL" },
  2: { bg: "bg-orange-50", border: "border-orange-300", badge: "bg-orange-500 text-white", icon: "TIER 2 URGENT" },
  3: { bg: "bg-blue-50", border: "border-blue-200", badge: "bg-blue-500 text-white", icon: "TIER 3 IMPORTANT" },
};

export default function ClinicalDiagnosisPanel({ diagnoses }: ClinicalDiagnosisPanelProps) {
  if (!diagnoses || diagnoses.length === 0) {
    return (
      <div className="rounded-lg p-4 bg-green-50 border border-green-200">
        <p className="text-sm font-semibold text-green-800">No clinical diagnoses triggered</p>
        <p className="text-xs text-green-600 mt-1">All pathology probabilities below diagnostic thresholds</p>
      </div>
    );
  }

  const grouped = new Map<number, ClinicalDiagnosis[]>();
  for (const dx of diagnoses) {
    const list = grouped.get(dx.tier) || [];
    list.push(dx);
    grouped.set(dx.tier, list);
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">Clinical Diagnoses</h3>
        <span className="text-xs text-gray-400">{diagnoses.length} condition{diagnoses.length !== 1 ? "s" : ""}</span>
      </div>

      {[1, 2, 3].map((tier) => {
        const items = grouped.get(tier);
        if (!items) return null;
        const style = TIER_STYLES[tier];

        return (
          <div key={tier} className="space-y-2">
            <span className={`inline-block text-xs font-bold px-2 py-0.5 rounded ${style.badge}`}>
              {style.icon}
            </span>
            {items.map((dx) => (
              <div key={dx.id} className={`rounded-lg p-3 border ${style.border} ${style.bg}`}>
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
                    <span key={code} className="text-xs bg-white/70 text-gray-600 px-1.5 py-0.5 rounded border border-gray-200">
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
      })}
    </div>
  );
}
