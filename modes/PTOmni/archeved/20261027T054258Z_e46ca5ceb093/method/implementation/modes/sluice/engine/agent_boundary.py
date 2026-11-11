"""Default-deny native Agent view, separate from profiling storage and controller.

The parent retains the only controller and public CLI connection. The child sees
Candidate source, public contracts, frozen code/knowledge and its own native
history. Every tool call synchronizes source through no-follow public copiers.
"""
from __future__ import annotations

import contextlib
import traceback
import os
from pathlib import Path
import shutil
import socket
import tempfile
import threading

from .controller import Controller, save
from .evidence_port import configured_evidence
from .knowledge_port import bound_knowledge, sha
from .tools_server import serve
from ..foundation.protocol import ProtocolError


class AgentBoundary:
    def __init__(self, context, config, *, controller=None):
        import isolation
        self.context, self.config = context, config
        self.temporary = tempfile.TemporaryDirectory(prefix='sluice-agent-')
        self.private = Path(self.temporary.name)
        self.workspaces = self.private / 'workspaces'
        self.public = self.private / 'public'
        self.workspaces.mkdir()
        self.public.mkdir()
        self.socket_path = self.private / 'tools.sock'
        self.server = socket.socket(socket.AF_UNIX)
        self.server.bind(str(self.socket_path))
        self.server.listen(8)
        self.server.settimeout(.2)
        self.cancel = threading.Event()
        self.connections = set()
        self.workers = []
        self.lock = threading.RLock()
        if controller is None:
            release_path = context.method_root / 'resources/knowledge_release.json'
            if sha(release_path) != config.get('knowledge_binding_sha256'):
                raise ProtocolError('Prepared knowledge binding mismatch')
            knowledge = bound_knowledge(
                context.method_root / 'resources/knowledge',
                validated_environment_digest=config.get('knowledge_environment'),
            )
            if knowledge.binding['knowledge_digest'] != config.get('knowledge_digest'):
                raise ProtocolError('Prepared knowledge package mismatch')
            controller = Controller(context, knowledge=knowledge,
                evidence=configured_evidence(context, config.get('evidence_port')), cancel=self.cancel)
        self.controller = controller
        self.cancel = self.controller.cancel
        self._export_workspaces()
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()
        self.isolation = isolation

    def _export_workspaces(self):
        import file_tree
        import workspace
        for original in self.context.workspace.parent.iterdir():
            if original.is_symlink() or not original.is_dir():
                continue
            try:
                self.context.validate_workspace(original)
            except ProtocolError:
                continue
            target = self.workspaces / original.name
            if target.exists():
                continue
            entries = file_tree.scan_tree(original, excluded_names=workspace.CANDIDATE_EXCLUDES, reject_special=False)
            if any(e['kind'] not in ('file', 'directory') for e in entries.values()):
                raise ProtocolError('Candidate source contains unsupported entries')
            target.mkdir()
            workspace._copy_source_tree(original, target, entries)
            public = self.public / original.name
            # The public context may itself be the broker's exact readonly link.
            shutil.copytree(self.context.public_context_root(original), public, symlinks=False)
            (target / '.benchmark_context').symlink_to(Path('/ptobench/public') / original.name)

    def _sync_source(self):
        import file_tree
        import workspace
        for source in self.workspaces.iterdir():
            target = self.context.validate_workspace(self.context.workspace.parent / source.name)
            entries = file_tree.scan_tree(source, excluded_names=workspace.CANDIDATE_EXCLUDES,
                                          reject_special=False)
            if any(e['kind'] not in ('file', 'directory') for e in entries.values()):
                raise ProtocolError('Candidate source contains unsupported entries')
            temporary = self.private / ('incoming-' + source.name)
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir()
            workspace._copy_source_tree(source, temporary, entries)
            if entries != file_tree.scan_tree(source, excluded_names=workspace.CANDIDATE_EXCLUDES,
                                              reject_special=False):
                raise ProtocolError('Candidate source changed while submitting')
            from ..foundation.source import fingerprint
            self.controller.validate_source(fingerprint(temporary))
            with file_tree.open_directory(target) as fd:
                for name in os.listdir(fd):
                    if name not in workspace.CANDIDATE_EXCLUDES:
                        file_tree.remove_at(fd, name)
            # The Agent's private /tmp and exported workspace can be on
            # different mounts; copy the pinned snapshot rather than rename.
            workspace._copy_source_tree(temporary, target, entries)
            shutil.rmtree(temporary)

    def call(self, *args):
        with self.lock:
            try:
                self._sync_source()
                result = self.controller.call(*args)
                self._export_workspaces()
                return result
            except Exception:
                save(self.context.work_root / 'agent_boundary_error.json', {'traceback': traceback.format_exc()})
                raise

    def context_projection(self):
        with self.lock:
            self._sync_source()
            return self.controller.context_projection()

    def record_context_sent(self, args):
        with self.lock:
            return self.controller.record_context_sent(args)

    def _accept(self):
        while not self.cancel.is_set():
            try:
                connection, _ = self.server.accept()
            except socket.timeout:
                continue
            self.connections.add(connection)
            worker = threading.Thread(target=self._serve, args=(connection,), daemon=True)
            self.workers.append(worker)
            worker.start()

    def _serve(self, connection):
        try:
            with connection, connection.makefile('r') as reader, connection.makefile('w') as writer:
                serve(self, reader, writer, cancel_on_disconnect=False)
        except (OSError, ValueError):
            pass
        finally:
            self.connections.discard(connection)

    def command(self, argv, environment):
        isolation = self.isolation
        view = isolation.View().system().native_agent()
        # Agent code is frozen/read-only; no method/work or capture mounts.
        for name in ('implementation', 'resources'):
            path = self.context.method_root / name
            if path.is_dir():
                view.add(path)
        for name in ('config.json', 'prompt.txt'):
            view.add(self.context.method_root / name)
        agent = self.context.session_root / 'agent'
        agent.mkdir(exist_ok=True)
        view.add(agent, write=True)
        directory = self.context.method_root / 'agent'
        directory.mkdir(exist_ok=True)
        view.add(directory, write=True)
        view.add(self.workspaces, self.context.workspace.parent, write=True)
        view.add(self.public, '/ptobench/public')
        view.add(self.socket_path, '/ptobench/sluice-tools.sock')
        # Nested execution can have a synthetic HOME; use the selected executable
        # and PATH rather than rediscovering a host Agent configuration.
        for item in environment.get('PATH', '').split(os.pathsep):
            path = Path(item)
            if path.is_absolute() and path.name == 'bin' and path.parent.name == 'npm':
                for child in ('bin', 'lib/node_modules'):
                    if (path.parent / child).exists():
                        view.add(path.parent / child)
        executable = Path(shutil.which(argv[0], path=environment.get('PATH')) or argv[0])
        if executable.is_absolute() and executable.is_file():
            if not any(Path(m[1]) == executable or Path(m[1]) in executable.parents for m in view.mounts):
                view.add(executable)
            # Installed CLI wrappers may import their package beside the target.
            actual = executable.resolve()
            if 'node_modules' in actual.parts:
                parent = actual
                while parent.name != 'node_modules':
                    parent = parent.parent
                view.add(parent)
        for argument in self.config['profile']['argv'][1:]:
            if Path(argument).is_absolute() and Path(argument).is_file():
                view.add(argument)
        # Keep only native Agent/profile settings. Never forward the outer
        # public-CLI broker socket or evaluator start descriptors.
        environment = {k: v for k, v in environment.items()
                       if k not in {'PTOBENCH_RPC_SOCKET', 'PTOBENCH_ACTION_START_FD'}}
        environment['SLUICE_TOOL_SOCKET'] = '/ptobench/sluice-tools.sock'
        save(self.context.method_root / 'agent/main/filesystem_view.json', view.manifest())
        return view.argv(argv, cwd=self.context.workspace, environment=environment)

    def close(self):
        self.cancel.set()
        for connection in tuple(self.connections):
            with contextlib.suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
        self.thread.join(timeout=1)
        for worker in self.workers:
            worker.join(timeout=25)
        self.server.close()
        evidence = self.controller.evidence
        if hasattr(evidence, 'db'):
            evidence.db.close()
        self.temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
