"""Revalidate retained capture bytes before projecting cached SQL evidence."""

KINDS = {
    'pmu': frozenset({'pmu', 'pmu_coverage_records'}),
    'compiler': frozenset({'pipeline_profile', 'perf_hints', 'memory',
                           'allocation_ir', 'compiler_hardware'}),
    'runtime': frozenset({'deps', 'chip_swimlane_records', 'dispatch_pairing', 'dispatch_identity'}),
    'clean': frozenset({'clean_result'}),
}
REQUIRED = {'pmu': {'pmu'}, 'runtime': {'deps', 'chip_swimlane_records'}, 'clean': {'clean_result'}}


def validate(db, handle):
    mode = handle.identity.profile_mode
    if mode == 'incore':
        raise ValueError('mode_unsupported')
    if mode not in KINDS:
        return
    from profile_db.errors import QueryError
    from profile_db.runtime_adapter.storage import resolve
    kinds = {r[0] for r in db.connection.execute(
        'SELECT kind FROM artifact WHERE run_id=?', [handle.db_run_id]).fetchall()} & KINDS[mode]
    reason = 'diagnostic_original_invalid' if mode in ('pmu', 'compiler') else 'capture_original_invalid'
    if not kinds or not REQUIRED.get(mode, set()) <= kinds:
        raise ValueError(reason)
    try:
        for kind in sorted(kinds):
            # Hash the bytes on every public read. Size/mtime alone cannot
            # detect replacements that retain the original filesystem stats.
            resolve(db.connection, handle.db_run_id, kind)
    except (QueryError, OSError, ValueError) as exc:
        # Internal locators and artifact names never enter public errors.
        raise ValueError(reason) from exc
