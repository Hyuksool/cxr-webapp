import { spawn } from "child_process";

export interface ClaudeCLIOptions {
  tools?: string[];
  model?: string;
  timeout?: number;
  addDirs?: string[];
}

/**
 * Call Claude via CLI in non-interactive mode.
 * Uses Claude Max subscription — no API key needed.
 * ANTHROPIC_API_KEY is explicitly removed from env to force Max auth.
 */
export async function queryClaudeCLI(
  prompt: string,
  options: ClaudeCLIOptions = {}
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

  // Remove ANTHROPIC_API_KEY to force Claude Max subscription auth
  const envCopy: Record<string, string | undefined> = { ...process.env, NO_COLOR: "1" };
  delete envCopy.ANTHROPIC_API_KEY;

  return new Promise((resolve, reject) => {
    const child = spawn("claude", args, {
      env: envCopy as NodeJS.ProcessEnv,
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
