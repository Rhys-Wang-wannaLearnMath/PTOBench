/** Persistent, Run-local SLUICE transport. A single controller owns all tool calls. */
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { createHash } from "node:crypto";

const CONTEXT_PREFIX = "<!-- SLUICE_ROUND_CONTEXT:";

function messageText(message: any): string {
  if (typeof message?.content === "string") return message.content;
  if (Array.isArray(message?.content)) {
    return message.content.filter((item: any) => item?.type === "text")
      .map((item: any) => item.text ?? "").join("\n");
  }
  return "";
}

function isSluiceContext(message: any): boolean {
  return message?.role === "user" && messageText(message).startsWith(CONTEXT_PREFIX);
}

function publicMessageManifest(messages: any[]) {
  const calls = messages.flatMap((message: any) => message?.role === "assistant" && Array.isArray(message.content)
    ? message.content.filter((item: any) => item?.type === "toolCall").map((item: any) => item.id)
    : []);
  const results = messages.filter((message: any) => message?.role === "toolResult")
    .map((message: any) => message.toolCallId);
  return {
    message_count: messages.length,
    conversation_sha256: createHash("sha256").update(JSON.stringify(messages)).digest("hex"),
    tool_call_ids: calls,
    tool_result_ids: results,
  };
}

function assertToolPairs(messages: any[]) {
  const pending = new Set<string>();
  for (const message of messages) {
    if (message?.role === "assistant" && Array.isArray(message.content)) {
      for (const item of message.content) {
        if (item?.type !== "toolCall") continue;
        if (typeof item.id !== "string" || pending.has(item.id)) throw new Error("Invalid SLUICE conversation tool call");
        pending.add(item.id);
      }
    } else if (message?.role === "toolResult") {
      if (typeof message.toolCallId !== "string" || !pending.delete(message.toolCallId)) {
        throw new Error("SLUICE conversation contains an unmatched tool result");
      }
    }
  }
  if (pending.size) throw new Error("SLUICE conversation contains an unfinished tool call");
}

function retainCurrentCycle(messages: any[]) {
  let boundary = -1;
  for (let index = 0; index < messages.length; index++) {
    const message = messages[index];
    if (message?.role !== "assistant" || !Array.isArray(message.content)) continue;
    if (message.content.some((item: any) => item?.type === "toolCall" && item.name === "sluice_step" &&
      typeof item.arguments?.completed_cycle_id === "string" && item.arguments.completed_cycle_id.length > 0)) {
      boundary = index;
    }
  }
  if (boundary < 0) return messages;
  return messages.filter((message, index) => index >= boundary || message?.role === "user");
}

export default function (pi: any) {
  const config = process.env.SLUICE_CONFIG;
  const helper = process.env.SLUICE_TOOLS;
  const python = process.env.SLUICE_PYTHON;
  if (!config || !helper || !python) throw new Error("SLUICE session configuration is missing");
  let child: ReturnType<typeof spawn> | undefined;
  let contextUnavailable = false;
  let seq = 0;
  const pending = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void }>();
  const call = (method: string, params: any = {}) => {
    if (!child) {
      child = spawn(python, [helper, "--config", config], { stdio: ["pipe", "pipe", "inherit"] });
      createInterface({ input: child.stdout! }).on("line", (line) => {
        const value = JSON.parse(line);
        const waiter = pending.get(value.id);
        pending.delete(value.id);
        if (value.error) waiter?.reject(new Error(value.error.message));
        else waiter?.resolve(value.result);
      });
      const fail = () => {
        for (const waiter of pending.values()) waiter.reject(new Error("SLUICE controller exited"));
        pending.clear();
      };
      child.on("error", fail);
      child.on("exit", fail);
    }
    return new Promise<any>((resolve, reject) => {
      const id = ++seq;
      pending.set(id, { resolve, reject });
      child!.stdin!.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    });
  };
  // Pi invokes session_start after loading extensions; registerTool accepts JSON schema objects.
  pi.on("session_start", async () => {
    await call("initialize", { protocolVersion: "2024-11-05" });
  });
  pi.on("context", async (event: any) => {
    try {
      const response = await call("sluice/context");
      const roundId = response?.round_id;
      const projection = response?.projection;
      if (!Array.isArray(event?.messages) || typeof roundId !== "string" || !projection || typeof projection.state_revision !== "string") {
        throw new Error("Invalid SLUICE context projection");
      }
      const transcript = event.messages.filter((message: any) => !isSluiceContext(message));
      assertToolPairs(transcript);
      const retained = retainCurrentCycle(transcript);
      assertToolPairs(retained);
      const contextMessage = CONTEXT_PREFIX + " " + roundId + " -->\n" +
        JSON.stringify(projection) + "\n<!-- /SLUICE_ROUND_CONTEXT -->";
      const messages = [...retained, { role: "user", content: contextMessage, timestamp: Date.now() }];
      await call("sluice/context_sent", {
        round_id: roundId,
        state_revision: projection.state_revision,
        message: contextMessage,
        message_manifest: publicMessageManifest(messages),
      });
      contextUnavailable = false;
      return { messages };
    } catch {
      contextUnavailable = true;
      const retained = Array.isArray(event?.messages)
        ? event.messages.filter((message: any) => message?.role === "user" && !isSluiceContext(message))
        : [];
      return { messages: [...retained, {
        role: "user",
        content: "SLUICE could not refresh its trusted current context for this request. Do not edit or run any optimization action until the context hook succeeds on a later request.",
        timestamp: Date.now(),
      }] };
    }
  });
  pi.on("session_shutdown", () => { child?.stdin?.end(); });
  // Discovery must happen during extension loading, before the first model request.
  return call("tools/list").then((listed) => {
    for (const spec of listed.tools) {
      pi.registerTool({
        name: spec.name, label: spec.name, description: spec.description, parameters: spec.inputSchema,
        async execute(toolCallId: string, params: any, signal: AbortSignal) {
          if (contextUnavailable && !["sluice_progress", "sluice_search", "sluice_read_skill", "sluice_profile_catalog", "sluice_profile_detail", "sluice_profile_raw"].includes(spec.name)) {
            throw new Error("SLUICE trusted context is unavailable; action tools are temporarily blocked");
          }
          if (signal?.aborted) throw new Error("Tool call cancelled before dispatch");
          const abort = () => child?.stdin?.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/cancelled", params: { reason: "Pi cancellation", toolCallId } }) + "\n");
          signal?.addEventListener("abort", abort, { once: true });
          try { return await call("tools/call", { name: spec.name, arguments: params, _meta: { toolCallId } }); }
          finally { signal?.removeEventListener("abort", abort); }
        },
      });
    }
  });
}
