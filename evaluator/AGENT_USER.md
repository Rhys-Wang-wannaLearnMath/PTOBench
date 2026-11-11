# Dedicated Agent account host procedure

This document records the one-time host contract for Plan 01. It is a
runbook for an administrator; the implementation does not create accounts,
change ACLs, or edit `/etc/sudoers`.

## Account and group

Inspect existing principals before making any account change:

```sh
getent passwd ptobench-agent
getent group ptobench-share
id ptobench-agent
passwd -S ptobench-agent
id <broker-user>
```

`ptobench-agent` must have a locked password, a non-interactive `nologin`
shell, UID/GID distinct from the Broker, and only `ptobench-share` as its
effective group. Add the Broker to `ptobench-share`, then start the queue from
a newly refreshed SSH or tmux login. Do not stop or clean older tmux sessions.

Only create a principal when `getent` confirms it is absent. If it exists but
does not meet the contract above, stop and have an administrator review it;
do not adopt or rewrite it. On a new host the creation commands are:

```sh
sudo groupadd --system ptobench-share
sudo useradd --system --gid ptobench-share --home-dir /nonexistent \
  --shell /usr/sbin/nologin ptobench-agent
sudo passwd --lock ptobench-agent
sudo usermod --append --groups ptobench-share <broker-user>
```

Replace `<broker-user>` with the verified queue owner. After changing group
membership, refresh the login before starting a queue.

Install the system Bubblewrap and ACL utilities, and use the system paths
resolved by `command -v`; do not use copies from HOME, npm, pip, or a
temporary directory.

```sh
command -v bwrap setfacl getfacl
bwrap --version
```

## Fixed sudo rule

Render and review [`ptobench-agent-user.sudoers.example`](ptobench-agent-user.sudoers.example)
against the host's `environment_lock.locked_python_executable()` and checkout
paths. The rule must be root-owned, mode `0440`, and pass:

```sh
sudo install -o root -g root -m 0440 evaluator/ptobench-agent-user.sudoers.example \
  /etc/sudoers.d/ptobench-agent-user
visudo -cf /etc/sudoers.d/ptobench-agent-user
sudo -l -U ptobench-agent
```

The only passwordless command is the exact locked Python `-I -S` invocation
of `evaluator/agent_dispatch.py`, running as
`ptobench-agent:ptobench-share`. It has `NOSETENV`, no command wildcards, no
Shell, and no additional Agent sudo permission. The command-specific
`env_keep` entry preserves only `PTOBENCH_AGENT_LAUNCH_REQUEST`; the dispatcher
deletes that variable before starting Bubblewrap.

Check execution rather than using `sudo -l` as an argument-matching test:

```sh
env -u PTOBENCH_AGENT_LAUNCH_REQUEST sudo -n -u ptobench-agent -g ptobench-share -- \
  /data1/home/ziruiwang/.local/share/uv/python/cpython-3.10.20-linux-aarch64-gnu/bin/python3.10 \
  -I -S /data1/home/ziruiwang/pypto-lib/benchmarks/PTOBench/evaluator/agent_dispatch.py
env -u PTOBENCH_AGENT_LAUNCH_REQUEST sudo -n -u ptobench-agent -g ptobench-share -- \
  /data1/home/ziruiwang/.local/share/uv/python/cpython-3.10.20-linux-aarch64-gnu/bin/python3.10 \
  -I -S /data1/home/ziruiwang/pypto-lib/benchmarks/PTOBench/evaluator/agent_dispatch.py unexpected-argument
```

Before the dispatcher exists, the first command may enter Python and fail
because the file is absent. The second command must not be authorized
passwordlessly. Once the dispatcher exists, record its fixed-entry canary,
stdin/stdout/stderr, signal and exit behavior, and the host-side Agent
identity. Do not modify an installed rule to accommodate another checkout.
For this implementation run, the active checkout is the canonical
`benchmarks/PTOBench` directory; do not add a parallel rule for the removed
Plan 01 worktree.

## ACL and mount review

Keep ACL changes minimal and path-specific. Review `getfacl -e` snapshots for
the full parent chain of the dispatcher, the approved `runs_root` to Session
workspace path, and every read/execute runtime path in the Agent mount
manifest. The dispatcher parents need Agent search (`--x`); Python, Node, Pi,
Bubblewrap, PyPTO runtime, and PTOBench transport directories may need `r-x`.
Do not grant access to the whole HOME, CANN/PTOAS/compiler trees, NPU device
nodes, or Candidate Worker paths. Do not recursively change ownership or
permissions. Bubblewrap mounts, not host listing permissions, define the
Agent's visible filesystem.

For this host the measured path contract is:

| Purpose | Host path and resolution | Required access |
|---|---|---|
| Dispatcher and parent chain | `/data1/home/ziruiwang/pypto-lib/benchmarks/PTOBench/evaluator/agent_dispatch.py`; parents from `/data1` through `evaluator/` | Agent gets search-only (`--x`) on private parent directories and read-only access to the Broker-owned dispatcher (`ptobench-share`, mode `0440`). The Broker HOME is not listable. |
| Locked Python | `/data1/home/ziruiwang/.local/share/uv/python/cpython-3.10.20-linux-aarch64-gnu/bin/python3.10`; Case venv entry `/data1/home/ziruiwang/pypto-lib/.venv-official/bin/python` resolves to it | Read/execute for the locked interpreter and its runtime directories. Use the Case venv entry for tests that need installed packages. |
| PyPTO runtime | `/data1/home/ziruiwang/pypto-lib/.venv-official/lib/python3.10/site-packages/pypto/`; `simpler_setup` assets at `/data1/home/ziruiwang/pypto-lib/.venv-official/lib/python3.10/site-packages/simpler_setup/_assets/` | Read-only through the explicit Python-prefix mount; upstream runtime source `/data1/home/ziruiwang/pypto-official-747e1e4b/runtime` is a Worker dependency, not an Agent source mount. |
| Node and Pi | Node entry `/data1/home/ziruiwang/.local/node/current/bin/node` resolves to `/data1/home/ziruiwang/.local/node/node-v22.23.1-linux-arm64/bin/node`; Pi entry `/data1/home/ziruiwang/.local/npm/bin/pi` resolves to `/data1/home/ziruiwang/.local/npm/lib/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js` | Read/execute on the resolved Node binary and read-only Pi bundle paths actually mounted by Bubblewrap. |
| Bubblewrap | `/usr/bin/bwrap` | System-owned read/execute; do not substitute a HOME, npm, pip, or temporary copy. |
| PTOBench CLI | Host source `/data1/home/ziruiwang/pypto-lib/benchmarks/PTOBench/evaluator/sandbox_client.py`; Broker copies it into the current Session and mounts that single file at `/data1/home/ziruiwang/pypto-lib/benchmarks/PTOBench/ptobench` inside the Agent view | The Agent sees only the per-Session transport copy, not the checkout's Evaluator, Launcher, Broker, modes, or run metadata. |
| Approved result root | `/data1/home/ziruiwang/pypto-lib/benchmarks/PTOBench/runs` | Existing path permissions suffice for host traversal; it is not mounted into the Agent view. Keep its ACL unchanged. |
| Dynamic Session workspace | `<runs_root>/<optimizer>/<case>/<session>/workspaces` | Broker-owned workspace entries; Candidate access is group-mediated, while Context and Outputs are read-only. The Session mount contains only the active workspace tree. |

The paths and resolved symlink targets above were inspected on 2026-11-11;
the package/system directories already provide the measured read/execute
access, and no additional ACL grant is needed. For another host or checkout,
resolve paths from its environment lock and generated mount manifest instead
of copying these values. Never grant the Agent CANN, PTOAS, compiler, or NPU
device access.

The canonical checkout's current ACL check found named `ptobench-agent`
entries with effective `--x` on both `benchmarks/PTOBench/` and
`benchmarks/PTOBench/evaluator/`. Their `other::r-x` entries remain, but the
named-user ACL takes precedence for the Agent. The launcher's effective-ACL
validation passed for the complete dispatcher parent chain. No ACL change is
needed; preserve the current ACLs.

The installed host rule is bound to the canonical
`/data1/home/ziruiwang/pypto-lib/benchmarks/PTOBench/evaluator/agent_dispatch.py`
path. Its no-request invocation entered the dispatcher and exited 2 with the
expected missing-request message; an extra-argument invocation was rejected
by `sudo -n`. The fixed-entry Bubblewrap canary also passed using a temporary
request. The Plan 01 worktree has been removed; use this canonical checkout.

## Acceptance boundary

Run the identity probe from a new Broker login and verify the Agent process
has only `ptobench-share`, zero effective capabilities, and no access to NPU
device nodes. Verify a Broker-owned shared Candidate canary is writable by
the Agent, a Broker-only file is unreadable, and a live Worker publication is
visible without making feedback writable. Do not create a real Pi Session,
load a model, or run a Case/NPU workload for this Plan 01 canary.
