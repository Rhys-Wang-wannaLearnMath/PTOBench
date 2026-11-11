# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public Python API (DESIGN.md 9) — the single source of truth.

The CLI (and later the MCP tools) wrap this module and render through the
same ``format_result``, so the ``facts`` format is byte-identical across
every entry point. ``ProfileDB`` is the connection manager with two
convenience methods layered on top: ``ingest`` and ``query`` (both drive
the milestone modules, never a parallel implementation).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from profile_db.db import ProfileDB as _ProfileDB
from profile_db.errors import QueryError, RenderError
from profile_db.facts import (
    Evidence,
    Fact,
    serialize_facts,
    truncate_facts,
    truncated_line,
)
from profile_db.query import execute as execute_query

DEFAULT_BUDGET_BYTES = 4096
_FORMATS = ("facts", "json", "markdown")


@dataclass(frozen=True)
class ImageRef:
    """A rendered image reference (populated by the T6 render layer)."""

    kind: str
    path: str


@dataclass(frozen=True)
class FileRef:
    """An HTML or Perfetto export reference."""

    kind: str
    path: str


@dataclass(frozen=True)
class Result:
    """One query answer: facts, image/file references, and the budget flag."""

    facts: tuple[Fact, ...]
    images: tuple[ImageRef, ...]
    truncated: bool
    files: tuple[FileRef, ...] = ()


class ProfileDB(_ProfileDB):
    """The public database handle: connection management plus ``ingest``,
    ``query``, and ``render``. ``ProfileDB.memory()`` keeps the in-memory
    working-set mode with identical schema, derived rows, and behavior."""

    def ingest(self, source, *, prune_after: bool = True, prune_keep: int = 3, **meta) -> dict[str, Any]:
        """Ingest one capture directory; returns the summary dict. By
        default the working set is pruned afterward (latest ``prune_keep``
        runs survive); pass ``prune_after=False`` to disable.

        ``**meta`` forwards to ``ingest.ingest_capture`` (program /
        platform / device_id / captured_at / notes / tags / rank_label /
        copy / git_* / bench_*).
        """
        from profile_db.ingest import ingest_capture

        report = ingest_capture(self, source, **meta)
        if prune_after and self.path is not None:
            self.prune(keep=prune_keep)
        return report

    def ingest_diagnostic(self, source, *, mode, program, identity, allowed_root=None, copy=False):
        from profile_db.ingest.diagnostic import ingest
        return ingest(self, source, mode=mode, program=program, identity=identity, allowed_root=allowed_root, copy=copy)

    def ingest_benchmark(self, source, *, program, identity, receipt, allowed_root=None, copy=False):
        from profile_db.ingest.benchmark import ingest
        return ingest(self, source, program=program, identity=identity, receipt=receipt, allowed_root=allowed_root, copy=copy)

    def compare_benchmarks(self, before, after, *, source_pair=None):
        from profile_db.ingest.benchmark import compare
        return compare(self, before, after, source_pair=source_pair)




    def repair(self, *, run_id: int, source=None, dry_run=False):
        from profile_db.ingest.repair import repair
        return repair(self, run_id=run_id, source=source, dry_run=dry_run)

    def repair_all(self, *, dry_run=False):
        """Repair stale physical or diagnostic captures; each run commits independently.

        Missing originals are reported per run. A failed run remains intact and
        does not prevent repair of other recoverable historical captures.
        """
        from profile_db.schema import CAPTURE_PARSER_VERSION, DIAGNOSTIC_PARSER_VERSION
        from profile_db.errors import PfdbError
        runs = [r[0] for r in self.connection.execute(
            "SELECT r.run_id FROM run r LEFT JOIN diagnostic_protocol d ON d.run_id=r.run_id "
            "WHERE (d.run_id IS NULL AND r.capture_parser_version IS DISTINCT FROM ?) "
            "OR (d.mode IN ('pmu','compiler') AND d.parser_version IS DISTINCT FROM ?) ORDER BY r.run_id",
            [CAPTURE_PARSER_VERSION, DIAGNOSTIC_PARSER_VERSION]).fetchall()]
        reports = []
        for run in runs:
            try:
                reports.append(self.repair(run_id=run, dry_run=dry_run))
            except (PfdbError, ValueError, OSError) as exc:
                reports.append(dict(run_id=run, status="failed", reason=str(exc)))
        return dict(ok=all(r["status"] != "failed" for r in reports), dry_run=dry_run, runs=reports)

    def ingest_incore(self, source, *, run_id: int) -> dict[str, Any]:
        """Attach an in-core collection (``manifest_export.csv``) to an
        existing run; delegates to generic existing-artifact attachment. Returns the summary."""
        from profile_db.ingest import ingest_incore

        return ingest_incore(self, source, run_id=run_id)

    def attach(self, source, *, run_id: int, copy: bool = False):
        from profile_db.ingest.analysis import attach

        return attach(self, source, run_id=run_id, copy=copy)

    def refresh(self, *, run_id: int, source=None):
        from profile_db.ingest.analysis import refresh

        return refresh(self, run_id=run_id, source=source)

    def query(self, name: str, *, budget_bytes: int | None = None, **params: Any) -> Result:
        """Run a registered query and return its Result envelope.

        ``budget_bytes=None`` uses the query's registered default.
        """
        output = execute_query(self.connection, name, params, budget_bytes=budget_bytes)
        return Result(facts=tuple(output.facts), images=(), truncated=output.truncated)

    def render(
        self,
        kind: str,
        run_id: int,
        *,
        render_dir: Path | str | None = None,
        t0_us: float | None = None,
        t1_us: float | None = None,
        task_id: str | None = None,
        core_index: int | None = None,
        image_max_bytes: int | None = None,
        cache_max_bytes: int | None = None,
        format: str = "png",
        kernel: str | None = None,
        run_ids: list[int] | None = None,
    ) -> Result:
        """Render one swimlane view (R0–R3) and return its Result.

        The image is written to the render cache (``<db>.pfdb/render`` by
        default, or ``render_dir``) and referenced through an ``ImageRef``
        plus an ``IMAGE`` fact carrying the manifest metadata — the text
        channel multimodal-agnostic models read without opening pixels.
        """
        from profile_db.render import render as render_image
        from profile_db.render.styles import (
            DEFAULT_CACHE_MAX_BYTES,
            DEFAULT_IMAGE_MAX_BYTES,
        )

        if render_dir is None:
            if self.path is None:
                raise RenderError("render_dir is required when rendering from an in-memory database")
            render_dir = self.path.parent / "render"
        if format != "png" or kind in ("deps", "scope", "host", "incore", "merged"):
            from profile_db.render.exports import render_export

            try:
                if task_id is not None:
                    from profile_db.task_ids import normalize_task_id

                    task_id = normalize_task_id(task_id).canonical
                meta = render_export(
                    self.connection,
                    run_id,
                    kind,
                    format,
                    render_dir,
                    dict(
                        t0_us=t0_us,
                        t1_us=t1_us,
                        task_id=task_id,
                        core_index=core_index,
                        kernel=kernel,
                        run_ids=run_ids,
                    ),
                    cache_max_bytes if cache_max_bytes is not None else DEFAULT_CACHE_MAX_BYTES,
                    image_max_bytes if image_max_bytes is not None else DEFAULT_CACHE_MAX_BYTES,
                )
            except (
                QueryError,
                ImportError,
                AttributeError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
                RuntimeError,
                OSError,
            ) as exc:
                return Result(
                    (Fact("FILE", dict(run_id=run_id, kind=kind, reason=str(exc)), Evidence.UNAVAILABLE),),
                    (),
                    False,
                )
            return Result(
                (Fact("FILE", meta, Evidence.MEASURED),), (), False, (FileRef(kind=kind, path=meta["path"]),)
            )
        rendered = render_image(
            self.connection,
            run_id,
            kind,
            render_dir=render_dir,
            t0_us=t0_us,
            t1_us=t1_us,
            task_id=task_id,
            core_index=core_index,
            image_max_bytes=image_max_bytes if image_max_bytes is not None else DEFAULT_IMAGE_MAX_BYTES,
            cache_max_bytes=cache_max_bytes if cache_max_bytes is not None else DEFAULT_CACHE_MAX_BYTES,
        )
        return _result_from_render(rendered)

    # -- lifecycle & short-term memory (DESIGN.md 8) ------------------------

    @contextmanager
    def _writing(self):
        """Every mutation outside ingest/prune goes through here.

        DESIGN.md 5.1 makes the database single-writer and puts mutual
        exclusion on the library lock. The trial/baseline/note writers
        assign surrogate ids as ``max(id) + 1``, which is a read-then-write
        that needs the lock to stay correct, and a transaction so a failed
        multi-statement write leaves no partial row.
        """
        from profile_db.db import WriterGuard

        with WriterGuard(self.path or ":memory:"):
            self.connection.execute("BEGIN TRANSACTION")
            try:
                yield self.connection
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
            self.connection.execute("COMMIT")

    def prune(self, keep: int = 3) -> dict[str, Any]:
        """Delete every run outside the working set (latest ``keep`` +
        baseline- and active-trial-referenced runs). Returns a report."""
        from profile_db.lifecycle import prune_runs

        return prune_runs(self.connection, self.path, keep)

    def note(self, run_id: int, text: str) -> None:
        """Set the free-text note on a run."""
        if self.connection.execute("SELECT 1 FROM run WHERE run_id = ?", [run_id]).fetchone() is None:
            raise QueryError(f"run {run_id} does not exist")
        with self._writing() as conn:
            conn.execute("UPDATE run SET notes = ? WHERE run_id = ?", [text, run_id])

    def compare(
        self,
        run_a: int,
        run_b: int,
        *,
        family: str | None = None,
        source_pair: tuple[str, str] | None = None,
    ) -> Result:
        """Neutral before/after comparison (compatibility-gated). Raises
        ``LifecycleError`` when the runs are not comparable.

        ``family`` is ``None`` (run-level only), ``"*"`` (every family),
        or a single family name.
        """
        from profile_db.lifecycle import compare_runs

        if any((self.get_profile_handle(r) or {}).get('identity') is not None for r in (run_a, run_b)):
            gate = self.compatibility(run_a, run_b, source_pair=source_pair)
            if not gate['compatible']:
                from profile_db.errors import LifecycleError
                raise LifecycleError('profile identity incompatible: ' + ', '.join(gate['reasons']))

        modes = {(self.get_profile_handle(r) or {}).get('identity', {}).get('profile_mode')
                 for r in (run_a, run_b)} if all((self.get_profile_handle(r) or {}).get('identity') for r in (run_a, run_b)) else set()
        if modes == {'clean'}:
            if family is not None:
                from profile_db.errors import LifecycleError
                raise LifecycleError('clean comparison does not support runtime families')
            decision = self.compare_benchmarks(run_a, run_b, source_pair=source_pair)
            fields = {k: decision[k] for k in ('comparison', 'delta_us', 'relative_change', 'interpretation') if k in decision}
            return Result((Fact('COMPARE', {'run_a': run_a, 'run_b': run_b,
                'profile_compatible': True, 'observed_samples': True}, Evidence.PROVEN),
                Fact('BENCH_COMPARISON', fields, Evidence.PROVEN)), (), False)
        if modes & {'pmu', 'compiler'}:
            if family is not None:
                from profile_db.errors import LifecycleError
                raise LifecycleError('diagnostic comparison does not support runtime families')
            values = [dict(self.connection.execute('SELECT metric, value FROM diagnostic_metric WHERE run_id=?', [r]).fetchall())
                      for r in (run_a, run_b)]
            return _compare_result(dict(run_a=run_a, run_b=run_b, profile_compatible=True, diagnostic=True,
                deltas=[dict(metric=k, before=values[0][k], after=values[1][k], delta=values[1][k]-values[0][k],
                             ratio=values[1][k]/values[0][k] if values[0][k] else None)
                        for k in sorted(values[0].keys() & values[1].keys())]))

        comparison = compare_runs(self.connection, run_a, run_b, family=family)
        return _compare_result(comparison)

    def get_profile_handle(self, run_id: int) -> dict | None:
        """Internal provenance API; callers must project before exposing to an agent."""
        from profile_db.identity import profile_handle
        return profile_handle(self.connection, run_id)

    def compatibility(self, run_a: int, run_b: int, *, source_pair=None) -> dict:
        """Strict identity and analysis gate with value-free reason codes."""
        from profile_db.identity import compatibility
        return compatibility(self.connection, run_a, run_b, source_pair=source_pair)

    def register_trial(
        self,
        goal: str,
        hypothesis: str,
        changed_files: Sequence[str] = (),
        parent_trial_id: int | None = None,
    ) -> int:
        from profile_db.lifecycle import register_trial as _register_trial

        with self._writing() as conn:
            return _register_trial(
                conn,
                goal=goal,
                hypothesis=hypothesis,
                changed_files=changed_files,
                parent_trial_id=parent_trial_id,
            )

    def bind_trial(self, trial_id: int, run_id: int) -> None:
        from profile_db.lifecycle import bind_trial as _bind_trial

        with self._writing() as conn:
            _bind_trial(conn, trial_id, run_id)

    def attach_bench(self, trial_id: int, bench: Mapping[str, Any]) -> None:
        from profile_db.lifecycle import attach_bench as _attach_bench

        with self._writing() as conn:
            _attach_bench(conn, trial_id, bench)

    def set_verdict(self, trial_id: int, verdict: str, evidence_refs: Sequence[Any] = ()) -> None:
        from profile_db.lifecycle import set_verdict as _set_verdict

        with self._writing() as conn:
            _set_verdict(conn, trial_id, verdict, evidence_refs)

    def list_trials(self, *, active_only: bool = False) -> Result:
        from profile_db.lifecycle import list_trials as _list_trials

        facts = tuple(
            Fact(
                "TRIAL",
                {
                    k: v
                    for k, v in {
                        "trial_id": t["trial_id"],
                        "parent_trial_id": t["parent_trial_id"],
                        "run_id": t["run_id"],
                        "goal": t["goal"],
                        "hypothesis": t["hypothesis"],
                        "changed_files": t["changed_files"],
                        "status": t["status"],
                        "verdict": t["verdict"],
                        "evidence_refs": t["evidence_refs"],
                        "bench_mean_us": t.get("bench_mean_us"),
                        "bench_rounds": t.get("bench_rounds"),
                    }.items()
                    if v is not None
                },
                Evidence.MEASURED,
            )
            for t in _list_trials(self.connection, active_only=active_only)
        )
        return Result(facts=facts, images=(), truncated=False)

    def baseline_add(
        self,
        name: str,
        run_id: int,
        bench_mean_us: float | None = None,
        criteria: Mapping[str, Any] | None = None,
    ) -> int:
        from profile_db.lifecycle import add_baseline

        with self._writing() as conn:
            return add_baseline(
                conn,
                name=name,
                run_id=run_id,
                bench_mean_us=bench_mean_us,
                criteria=criteria,
            )

    def baseline_list(self) -> Result:
        from profile_db.lifecycle import list_baselines

        facts = tuple(
            Fact(
                "BASELINE",
                {
                    k: v
                    for k, v in {
                        "baseline_id": b["baseline_id"],
                        "name": b["name"],
                        "program": b["program"],
                        "platform": b["platform"],
                        "run_id": b["run_id"],
                        "bench_mean_us": b["bench_mean_us"],
                        "criteria": b["criteria"],
                    }.items()
                    if v is not None
                },
                Evidence.MEASURED,
            )
            for b in list_baselines(self.connection)
        )
        return Result(facts=facts, images=(), truncated=False)

    def baseline_diff(
        self,
        run_id: int,
        baseline_name: str | None = None,
    ) -> Result:
        from profile_db.lifecycle import diff_baseline

        comparison = diff_baseline(self.connection, run_id, baseline_name)
        return _compare_result(comparison)


def format_result(result: Result, fmt: str, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> str:
    """Render a Result in one of ``facts`` (DSL, byte-identical to the
    engine), ``json``, or ``markdown``.

    Every format is bounded by the same budget: the fact list is
    prefix-truncated by ``facts.truncate_facts`` first, then serialized,
    and each format carries its own explicit truncation marker. A format
    that silently emitted the untruncated list would let an agent bypass
    the budget by asking for ``json``.
    """
    if fmt not in _FORMATS:
        raise QueryError(f"unknown format {fmt!r}; use one of: {', '.join(_FORMATS)}")
    if fmt == "facts":
        return serialize_facts(result.facts, budget_bytes)
    kept, dropped = truncate_facts(result.facts, budget_bytes)
    if fmt == "json":
        payload = [_fact_dict(fact) for fact in kept]
        if dropped:
            payload.append(
                {
                    "rec": "TRUNCATED",
                    "first_dropped_index": len(kept),
                    "remaining": dropped,
                    "limit": budget_bytes,
                    "evidence": Evidence.UNAVAILABLE.value,
                }
            )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    text = _markdown(kept)
    if dropped:
        text += "\n" + truncated_line(dropped, len(kept), budget_bytes)
    return text


def _fact_dict(fact: Fact) -> dict[str, Any]:
    return {"rec": fact.rec, **dict(sorted(fact.fields.items())), "evidence": fact.evidence.value}


def _result_from_render(rendered) -> Result:
    """Turn a render-layer ``RenderResult`` into the public ``Result``
    envelope: an ``IMAGE`` fact (manifest metadata for the text channel)
    and, on success, an ``ImageRef`` for the multimodal channel."""
    if rendered.unavailable:
        fact = Fact(
            "IMAGE",
            {"run_id": rendered.run_id, "kind": rendered.kind, "reason": rendered.note},
            Evidence.UNAVAILABLE,
        )
        return Result(facts=(fact,), images=(), truncated=False)
    manifest = rendered.manifest
    x0, x1 = manifest["x_axis_us"]
    fields: dict[str, Any] = {
        "run_id": rendered.run_id,
        "kind": rendered.kind,
        "path": str(rendered.image_path),
        "sha256": rendered.sha256,
        "size_bytes": rendered.size_bytes,
        "width": manifest.get("width"),
        "height": manifest.get("height"),
        "x0_us": x0,
        "x1_us": x1,
        "us_per_px": manifest.get("us_per_px"),
        "downsampled": manifest.get("downsampled"),
        "cache_hit": rendered.cache_hit,
        "wall_ms": rendered.wall_ms,
        # Engine -> color mapping so text-channel consumers can interpret
        # the colors of an image they cannot (or do not) open.
        "legend": manifest.get("legend"),
    }
    if rendered.note is not None:
        fields["note"] = rendered.note
    fact = Fact("IMAGE", {k: v for k, v in fields.items() if v is not None}, Evidence.MEASURED)
    image = ImageRef(kind=rendered.kind, path=str(rendered.image_path))
    return Result(facts=(fact,), images=(image,), truncated=False)


def _num(value: Any) -> Any:
    """Display-round a float to nanosecond precision; ints and non-numbers
    pass through (mirrors the query layer's ``us`` for the DSL output)."""
    if value is None or isinstance(value, bool):
        return value
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return value


def _compare_result(comparison: Mapping[str, Any]) -> Result:
    """Turn a lifecycle comparison dict into COMPARE/DELTA facts."""
    number = (lambda value: value) if comparison.get("diagnostic") else _num
    header_fields = {
        "run_a": comparison["run_a"],
        "run_b": comparison["run_b"],
        "program": comparison.get("program"),
        "compatible": comparison.get("compatible", True),
        "profile_compatible": comparison.get("profile_compatible", True),
        "reasons": comparison.get("reasons", []),
    }
    for key in ("baseline", "baseline_run_id"):
        if key in comparison:
            header_fields[key] = comparison[key]
    facts: list[Fact] = [
        Fact("COMPARE", {k: v for k, v in header_fields.items() if v is not None}, Evidence.MEASURED),
        Fact(
            "NOTE",
            {
                "topic": "metric-scope",
                "bench_mean_us": "unprofiled",
                "makespan_us": "profiled-with-observer",
            },
            Evidence.PROVEN,
        ),
    ]
    for delta in comparison["deltas"]:
        fields: dict[str, Any] = {
            "run_a": comparison["run_a"],
            "run_b": comparison["run_b"],
            "metric": delta["metric"],
            "before": number(delta["before"]),
            "after": number(delta["after"]),
            "delta": number(delta["delta"]),
        }
        if delta.get("ratio") is not None:
            fields["ratio"] = number(delta["ratio"])
        if delta.get("family") is not None:
            fields["family"] = delta["family"]
        if "baseline" in comparison:
            fields["baseline"] = comparison["baseline"]
        facts.append(Fact("DELTA", fields, Evidence.MEASURED))
    return Result(facts=tuple(facts), images=(), truncated=False)


def _markdown(facts: Sequence[Fact]) -> str:
    lines = ["| record | fields | evidence |", "|---|---|---|"]
    for fact in facts:
        fields_text = " ".join(f"{k}={_json(v)}" for k, v in sorted(fact.fields.items()))
        lines.append(f"| {fact.rec} | {fields_text} | {fact.evidence.value} |")
    return "\n".join(lines)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
