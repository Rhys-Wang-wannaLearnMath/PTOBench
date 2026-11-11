#!/usr/bin/env python3
"""Run-local newline JSON-RPC stdio MCP transport; no global configuration writes."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "evaluator")]
from modes.sluice.engine.controller import Controller
from modes.sluice.engine.tools import specifications
from modes.sluice.foundation.context import RunContext
from modes.sluice.foundation.protocol import ProtocolError


def serve(controller, reader=sys.stdin, writer=sys.stdout, *, cancel_on_disconnect=True):
    send_lock = threading.Lock()
    workers = []

    def send(value):
        with send_lock:
            writer.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            writer.flush()

    def execute(request):
        identity = request["id"]
        params = request.get("params", {})
        try:
            result = controller.call(params["name"], params.get("arguments", {}), identity, request)
            from modes.sluice.engine.public_output import project
            result = project(result)
            value = {
                "content": [
                    {"type": "text", "text": json.dumps(result, ensure_ascii=False, allow_nan=False)}
                ],
                "isError": result["is_error"],
            }
            send({"jsonrpc": "2.0", "id": identity, "result": value})
        except Exception:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": identity,
                    "error": {
                        "code": -32603,
                        "message": "SLUICE tool execution failed; inspect method records",
                    },
                }
            )

    try:
        for line in reader:
            try:
                if len(line) > 131072:
                    raise ValueError()
                request = json.loads(line)
                if type(request) is not dict:
                    raise ValueError()
            except ValueError:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Invalid JSON-RPC request"},
                    }
                )
                continue
            method = request.get("method")
            identity = request.get("id")
            params = request.get("params", {})
            if method == "notifications/cancelled":
                # One active action per session. Cancellation closes execution, not just its response.
                controller.cancel.set()
                continue
            if identity is None:
                continue
            if method == "initialize":
                result = {
                    "protocolVersion": request.get("params", {}).get("protocolVersion", "2024-11-05"),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "sluice", "version": "1.0.0"},
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": specifications()}
            elif method == "sluice/context":
                try:
                    result = controller.context_projection()
                except Exception:
                    send({
                        "jsonrpc": "2.0", "id": identity,
                        "error": {"code": -32000, "message": "SLUICE context projection is unavailable"},
                    })
                    continue
            elif method == "sluice/context_sent":
                try:
                    result = controller.record_context_sent(params)
                except Exception:
                    send({
                        "jsonrpc": "2.0", "id": identity,
                        "error": {"code": -32000, "message": "SLUICE context send record was rejected"},
                    })
                    continue
            elif method == "tools/call":
                worker = threading.Thread(target=execute, args=(request,))
                workers.append(worker)
                worker.start()
                continue
            else:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": identity,
                        "error": {"code": -32601, "message": "Method not found"},
                    }
                )
                continue
            send({"jsonrpc": "2.0", "id": identity, "result": result})
    finally:
        if cancel_on_disconnect:
            controller.cancel.set()
        for worker in workers:
            worker.join(timeout=25)


def main():
    parser = argparse.ArgumentParser(description="Expose the SLUICE tools over local stdio MCP.")
    parser.add_argument("--config", required=True, help="Prepared method configuration path.")
    args = parser.parse_args()
    initial_config = json.loads(Path(args.config).read_text())
    context = RunContext.from_request({"context": initial_config["context"]})
    if Path(args.config) != context.method_root / "config.json":
        raise ValueError("Config must bind to the prepared Run")
    from modes.sluice.engine.lifecycle import load_config
    from modes.sluice.engine.knowledge_port import bound_knowledge
    config = load_config(context)
    from modes.sluice.engine.evidence_port import configured_evidence
    knowledge = bound_knowledge(context.method_root / 'resources/knowledge',
        validated_environment_digest=config.get('knowledge_environment'))
    controller = Controller(context, knowledge=knowledge,
        evidence=configured_evidence(context, config.get('evidence_port')))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: controller.cancel.set())
    serve(controller)


if __name__ == "__main__":
    main()
