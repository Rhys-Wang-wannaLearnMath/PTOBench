/** Run-local CANNBot tools. No global Pi extension or skill installation. */
import { spawn, spawnSync } from "node:child_process";

export default function (pi: any) {
  const helper = process.env.CANNBOT_TOOLS;
  const python = process.env.CANNBOT_PYTHON;
  const config = process.env.CANNBOT_CONFIG;
  const dispatch = process.env.CANNBOT_DISPATCH;
  if (!helper || !python || !config || !dispatch) throw new Error("CANNBot session configuration is missing");
  const options = ["--config", config, "--dispatch", dispatch];
  if (process.env.CANNBOT_ALLOW_TASK === "1") options.push("--allow-task");
  const listed = spawnSync(python, [helper, "list", ...options], { encoding: "utf8" });
  if (listed.status !== 0) throw new Error(listed.stderr || "CANNBot tool discovery failed");
  for (const spec of JSON.parse(listed.stdout)) {
    pi.registerTool({
      name: spec.name, label: spec.name, description: spec.description, parameters: spec.inputSchema,
      async execute(toolCallId: string, params: any, signal: AbortSignal) {
        return await new Promise((resolve, reject) => {
          if (signal?.aborted) {
            resolve({ content: [{ type: "text", text: "Tool call cancelled before dispatch" }], isError: true });
            return;
          }
          const child = spawn(python, [helper, "call", ...options], { stdio: ["pipe", "pipe", "pipe"] });
          const out: Buffer[] = [], err: Buffer[] = [];
          let escalation: ReturnType<typeof setTimeout> | undefined;
          const abort = () => {
            child.kill("SIGTERM");
            escalation = setTimeout(() => child.kill("SIGKILL"), 4000);
          };
          signal?.addEventListener("abort", abort, { once: true });
          child.stdout.on("data", (chunk: Buffer) => out.push(chunk));
          child.stderr.on("data", (chunk: Buffer) => err.push(chunk));
          child.on("error", reject);
          child.on("close", (code) => {
            signal?.removeEventListener("abort", abort);
            if (escalation) clearTimeout(escalation);
            try {
              const value = JSON.parse(Buffer.concat(out).toString("utf8"));
              resolve(value);
            } catch {
              resolve({ content: [{ type: "text", text: Buffer.concat(err).toString("utf8") || `CANNBot helper exited (${code}) without a tool result` }], isError: true });
            }
          });
          child.stdin.on("error", () => {});
          child.stdin.end(JSON.stringify({ name: spec.name, arguments: params, tool_call_id: toolCallId }));
        });
      },
    });
  }
}
