import { createConnection } from "node:net";
import { readFileSync } from "node:fs";

function request(value: Record<string, unknown>): Promise<any> {
  const path = process.env.PTOBENCH_PI_TOOL_SOCKET;
  if (!path) return Promise.reject(new Error("PTOAnalyzer socket is unavailable"));
  return new Promise((resolve, reject) => {
    const socket = createConnection({ path });
    socket.setEncoding("utf8");
    let response = "", complete = false;
    socket.once("connect", () => socket.write(JSON.stringify(value) + "\n"));
    socket.on("data", (chunk: string) => {
      response += chunk;
      const newline = response.indexOf("\n");
      if (newline < 0) return;
      complete = true;
      socket.end();
      try { resolve(JSON.parse(response.slice(0, newline))); }
      catch (error) { reject(error); }
    });
    socket.once("error", reject);
    socket.once("close", () => { if (!complete) reject(new Error("PTOAnalyzer connection closed without a result")); });
  });
}

export default function (pi: any) {
  const specs = JSON.parse(readFileSync(new URL("./pto-analyzer-specs.json", import.meta.url), "utf8"));
  for (const spec of specs) {
    pi.registerTool({
      ...spec,
      async execute(toolCallId: string, params: Record<string, unknown>, signal?: AbortSignal) {
        let dispatched = false, cancelled = !!signal?.aborted;
        const cancel = () => {
          cancelled = true;
          if (dispatched) void request({ action: "cancel", tool_call_id: toolCallId }).catch(() => {});
        };
        signal?.addEventListener("abort", cancel, { once: true });
        try {
          // Send even a pre-cancelled invocation so the Broker can audit it.
          const response = request({ action: "call", tool_call_id: toolCallId, name: spec.name,
                                     arguments: params, cancelled });
          dispatched = true;
          return await response;
        } finally {
          signal?.removeEventListener("abort", cancel);
        }
      },
    });
  }
  pi.on("session_start", async () => {
    const names = pi.getAllTools().map((tool: any) => tool.name);
    const active_names = pi.getActiveTools();
    const result = await request({ action: "registered", names, active_names });
    if (result.isError) throw new Error(result.content[0].text);
  });
}
