import { NextRequest, NextResponse } from "next/server";

// Allow up to 5 minutes for Claude CLI report generation
export const maxDuration = 300;

const BACKEND_URL = process.env.CXR_BACKEND_URL || "http://localhost:8200";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const response = await fetch(`${BACKEND_URL}/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(300_000), // 5 min — Claude CLI can take 90-180s
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, error: `Backend error: ${response.status}` },
        { status: 500 }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      { success: false, error: message },
      { status: 500 }
    );
  }
}
