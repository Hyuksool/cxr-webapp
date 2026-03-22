import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.CXR_BACKEND_URL || "http://localhost:8200";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("cxr_image");

    if (!file || !(file instanceof Blob)) {
      return NextResponse.json(
        { success: false, error: "No image file provided" },
        { status: 400 }
      );
    }

    // Forward to Python backend
    const backendForm = new FormData();
    backendForm.append("cxr_image", file);

    const response = await fetch(`${BACKEND_URL}/analyze`, {
      method: "POST",
      body: backendForm,
      signal: AbortSignal.timeout(60_000),
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
