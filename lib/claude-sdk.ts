import { spawn } from "child_process";
import type Anthropic from "@anthropic-ai/sdk";

export interface ClaudeCLIOptions {
  tools?: string[];
  model?: string;
  timeout?: number;
  addDirs?: string[];
}

/**
 * Call Claude via Anthropic API (preferred) or CLI (fallback).
 *
 * If ANTHROPIC_API_KEY is set → uses SDK directly (~2-3s).
 * Otherwise → spawns `claude` CLI process (~15-20s).
 */
export async function queryClaudeCLI(
  prompt: string,
  options: ClaudeCLIOptions = {}
): Promise<string> {
  const apiKey = process.env.ANTHROPIC_API_KEY;

  if (apiKey) {
    return queryAnthropicAPI(prompt, options, apiKey);
  }

  return queryViaCLI(prompt, options);
}

/**
 * Direct Anthropic API call via SDK. Fast (~2-3s).
 */
async function queryAnthropicAPI(
  prompt: string,
  options: ClaudeCLIOptions,
  apiKey: string
): Promise<string> {
  const Anthropic = (await import("@anthropic-ai/sdk")).default;
  const client = new Anthropic({ apiKey });

  const modelMap: Record<string, string> = {
    haiku: "claude-haiku-4-5-20251001",
    sonnet: "claude-sonnet-4-6-20250311",
    opus: "claude-opus-4-6-20250311",
  };
  const model = modelMap[options.model || "haiku"] || modelMap.haiku;

  const message = await client.messages.create({
    model,
    max_tokens: 4096,
    messages: [{ role: "user", content: prompt }],
  });

  const textBlocks = message.content.filter((b) => b.type === "text") as Array<{ type: "text"; text: string }>;
  const text = textBlocks.map((b) => b.text).join("");

  return text.trim();
}

/**
 * Fallback: Claude CLI in non-interactive mode.
 * Uses the existing Claude Code subscription — no API key needed.
 */
function queryViaCLI(
  prompt: string,
  options: ClaudeCLIOptions
): Promise<string> {
  const args = [
    "-p",
    prompt,
    "--output-format",
    "text",
    "--no-session-persistence",
    "--dangerously-skip-permissions",
  ];

  if (options.tools !== undefined) {
    args.push("--tools", options.tools.join(","));
  }

  if (options.model) {
    args.push("--model", options.model);
  }

  if (options.addDirs) {
    args.push("--add-dir", ...options.addDirs);
  }

  return new Promise((resolve, reject) => {
    const child = spawn("claude", args, {
      env: { ...process.env, NO_COLOR: "1" },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (data: Buffer) => {
      stdout += data.toString();
    });

    child.stderr.on("data", (data: Buffer) => {
      stderr += data.toString();
    });

    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`Claude CLI timed out after ${options.timeout || 120_000}ms`));
    }, options.timeout || 120_000);

    child.on("close", (code: number | null) => {
      clearTimeout(timer);
      if (stdout.trim()) {
        resolve(stdout.trim());
      } else if (code !== 0) {
        reject(
          new Error(
            `Claude CLI exited with code ${code}${stderr ? `\nstderr: ${stderr}` : ""}`
          )
        );
      } else {
        resolve(stdout.trim());
      }
    });

    child.on("error", (err: Error) => {
      clearTimeout(timer);
      reject(new Error(`Claude CLI spawn error: ${err.message}`));
    });
  });
}

/**
 * Extract JSON from Claude's text response.
 * Strips markdown code fences and finds the first valid JSON object.
 */
export function extractJSON<T>(text: string): T {
  // Remove markdown code fences
  let cleaned = text.replace(/```json?\n?/g, "").replace(/```\n?/g, "").trim();

  // Try parsing the whole thing first
  try {
    return JSON.parse(cleaned);
  } catch {
    // Find the first { ... } block
    const start = cleaned.indexOf("{");
    const end = cleaned.lastIndexOf("}");
    if (start !== -1 && end !== -1 && end > start) {
      cleaned = cleaned.substring(start, end + 1);
      return JSON.parse(cleaned);
    }
    throw new Error(`Failed to extract JSON from response: ${text.substring(0, 300)}`);
  }
}
