"""Synthetic backend transport probe. Does not edit Candidate or spend hardware budget."""

import json
import os
import subprocess


def main():
    child = subprocess.Popen(
        [os.environ["SLUICE_PYTHON"], os.environ["SLUICE_TOOLS"], "--config", os.environ["SLUICE_CONFIG"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    for i, (method, params) in enumerate(
        [
            ("initialize", {}),
            ("tools/list", {}),
            ("tools/call", {"name": "sluice_progress", "arguments": {}}),
            (
                "tools/call",
                {"name": "sluice_finish", "arguments": {"reason": "Synthetic transport probe completed"}},
            ),
        ],
        1,
    ):
        request = {"jsonrpc": "2.0", "id": i, "method": method, "params": params}
        child.stdin.write(json.dumps(request) + "\n")
        child.stdin.flush()
        result = json.loads(child.stdout.readline())
        print(
            json.dumps({"type": "synthetic_tool_event", "request": request, "response": result}), flush=True
        )
    child.stdin.close()
    child.wait(timeout=30)
    print(json.dumps({"type": "synthetic_answer", "text": "Synthetic transport probe completed"}))


if __name__ == "__main__":
    main()
