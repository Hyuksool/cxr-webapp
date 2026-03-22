export interface PathologyResult {
  name: string;
  probability: number;
  urgency: "critical" | "urgent" | "routine" | "info" | "normal";
}

export interface ZeroShotPathology {
  name: string;
  similarity: number;
  urgency: "critical" | "urgent" | "routine" | "info" | "normal";
}

export interface ZeroShotResult {
  available: boolean;
  pathologies: ZeroShotPathology[];
  findings: ZeroShotPathology[];
  model: string;
}

export interface CXRAnalysisResult {
  pathologies: PathologyResult[];
  findings: PathologyResult[];
  urgency_level: "critical" | "urgent" | "routine" | "normal";
  heatmap_base64: string | null;
  model_pathologies: string[];
  confidence_score: number;
  no_finding_probability: number;
  zero_shot: ZeroShotResult | null;
}

export interface ReportResult {
  report: string;
  sections: Record<string, string>;
  urgency_level: string;
  model: string;
}
