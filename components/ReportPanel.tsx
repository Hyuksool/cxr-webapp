"use client";

import type { ReportResult } from "@/types/cxr";

interface ReportPanelProps {
  report: ReportResult;
}

// Sections to show first (prioritized order)
const PRIORITY_SECTIONS = ["PRIMARY DIAGNOSIS", "IMPRESSION", "CLINICAL IMPRESSION (AI)"];

export default function ReportPanel({ report }: ReportPanelProps) {
  const sections = report.sections || {};
  const hasSections = Object.keys(sections).length > 0;

  // Separate priority sections from the rest
  const prioritySections = PRIORITY_SECTIONS
    .filter(s => sections[s])
    .map(s => [s, sections[s]] as [string, string]);
  const otherSections = Object.entries(sections)
    .filter(([name]) => !PRIORITY_SECTIONS.includes(name));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">Radiology Report</h3>
        <span className="text-xs text-gray-400">{report.model}</span>
      </div>

      {hasSections ? (
        <div className="space-y-3">
          {/* Priority sections with highlight */}
          {prioritySections.map(([sectionName, content]) => (
            <div key={sectionName} className="rounded-lg border-2 border-blue-300 overflow-hidden bg-blue-50">
              <div className="bg-blue-100 px-3 py-2 border-b border-blue-200">
                <span className="text-xs font-bold text-blue-800 uppercase tracking-wide">
                  {sectionName}
                </span>
              </div>
              <div className="px-3 py-3">
                <p className="text-sm text-gray-800 whitespace-pre-wrap leading-relaxed font-medium">
                  {content}
                </p>
              </div>
            </div>
          ))}

          {/* Other sections */}
          {otherSections.map(([sectionName, content]) => (
            <div key={sectionName} className="rounded-lg border border-gray-200 overflow-hidden">
              <div className="bg-gray-50 px-3 py-2 border-b border-gray-200">
                <span className="text-xs font-bold text-gray-600 uppercase tracking-wide">
                  {sectionName}
                </span>
              </div>
              <div className="px-3 py-2">
                <p className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
                  {content}
                </p>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-gray-50 rounded-lg p-4">
          <p className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed font-mono">
            {report.report}
          </p>
        </div>
      )}

      <p className="text-xs text-gray-400 italic">
        AI-assisted report. Not a substitute for clinical judgment or board-certified radiologist review.
      </p>
    </div>
  );
}
