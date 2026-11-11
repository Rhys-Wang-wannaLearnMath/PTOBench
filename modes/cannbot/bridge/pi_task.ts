import { createConnection } from "node:net";

type ToolSpec = {
  name: string;
  label: string;
  description: string;
  parameters: Record<string, unknown>;
};

function schema(properties: Record<string, unknown>, required: string[] = []) {
  return { type: "object", properties, required, additionalProperties: true };
}

function request(value: Record<string, unknown>): Promise<any> {
  const path = process.env.PTOBENCH_CANNBOT_SOCKET;
  if (!path) return Promise.reject(new Error("CANNBot tool socket is unavailable"));
  return new Promise((resolve, reject) => {
    const socket = createConnection({ path });
    let response = "";
    socket.once("connect", () => socket.write(JSON.stringify(value) + "\n"));
    socket.on("data", (chunk: Buffer) => {
      response += chunk.toString("utf8");
      const newline = response.indexOf("\n");
      if (newline < 0) return;
      socket.end();
      try {
        resolve(JSON.parse(response.slice(0, newline)));
      } catch (error) {
        reject(error);
      }
    });
    socket.once("error", reject);
  });
}

export default function (pi: any) {
  const tools: ToolSpec[] = [
    {
      name: "skill",
      label: "skill",
      description: "Read an unchanged upstream skill and its source path.",
      parameters: schema({ name: { type: "string" } }, ["name"]),
    },
    {
      name: "todowrite",
      label: "todowrite",
      description: "Persist your Todo list unchanged in the method work directory.",
      parameters: schema({ todos: { type: "array", items: {} } }, ["todos"]),
    },
    {
      name: "todoread",
      label: "todoread",
      description: "Read your persisted Todo list.",
      parameters: schema({}),
    },
  ];
  if (process.env.PTOBENCH_CANNBOT_ALLOW_TASK === "1") {
    tools.unshift({
      name: "Task",
      label: "Task",
      description: "Dispatch an original optimizer or verifier role in a fresh native session. Sessions share source and PTOBench budget; the original answer or error is returned.",
      parameters: schema({
        subagent_type: { type: "string", enum: ["pypto-op-optimizer", "pypto-op-verifier"] },
        prompt: { type: "string" },
        description: { type: "string" },
        workspace: { type: "string" },
      }, ["subagent_type", "prompt"]),
    });
  }

  for (const spec of tools) {
    pi.registerTool({
      ...spec,
      async execute(toolCallId: string, params: Record<string, unknown>, signal: AbortSignal) {
        if (signal?.aborted) {
          return { content: [{ type: "text", text: "Tool call cancelled before dispatch" }], isError: true };
        }
        const cancel = () => {
          void request({ action: "cancel", tool_call_id: toolCallId }).catch(() => {});
        };
        signal?.addEventListener("abort", cancel, { once: true });
        try {
          return await request({
            action: "call",
            tool_call_id: toolCallId,
            name: spec.name,
            arguments: params,
          });
        } finally {
          signal?.removeEventListener("abort", cancel);
        }
      },
    });
  }
}
