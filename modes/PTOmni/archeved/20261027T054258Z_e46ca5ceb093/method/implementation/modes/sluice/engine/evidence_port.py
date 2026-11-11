"""Question-scoped ProfileDB facade. No raw capture parsing or MCP path."""
from __future__ import annotations
from dataclasses import replace

import math

from .contracts import EvidenceBundle, EvidenceFact
from .evidence_guard import encoded, opaque
from .profile_questions import QUESTIONS


def bounded(bundle, budget_bytes=4096, max_facts=16):
    if type(budget_bytes) is not int or budget_bytes < 768 or budget_bytes > 4096:
        raise ValueError('evidence_budget_invalid')
    facts = bundle.facts[:max_facts]
    missing_reasons = set(bundle.missing)
    truncated = bundle.truncated or len(facts) != len(bundle.facts)
    while True:
        missing = tuple(sorted(missing_reasons | ({'evidence_truncated'} if truncated else set())))
        value = replace(bundle, facts=facts, missing=missing, truncated=truncated,
                        comparison=bundle.comparison if facts or bundle.comparison == 'incomparable' else 'unavailable',
                        status='available' if any(f.status != 'unavailable' for f in facts) else 'unavailable')
        if len(encoded(value.to_dict())) <= budget_bytes:
            return value
        if not facts:
            if len(missing_reasons) > 1:
                # A small requested envelope must retain the refusal verdict.
                # Signal truncation so the agent can retry with a larger budget.
                important = {'physical_device_changed', 'physical_device_identity_unavailable',
                             'physical_device_mapping_changed', 'capture_original_invalid',
                             'diagnostic_original_invalid', 'device_identity_original_invalid'}
                remove = max(missing_reasons, key=lambda r: (r not in important, len(r), r))
                missing_reasons.remove(remove)
                truncated = True
                continue
            raise ValueError('evidence_budget_invalid')
        facts = facts[:-1]
        truncated = True


class UnavailableEvidencePort:
    version = 'sluice-unavailable-evidence-v2'
    available = False

    def query(self, *, question, before, after, budget_bytes=4096):
        ref = opaque('evidence', question)
        return bounded(EvidenceBundle(question, (EvidenceFact(question, 'availability', None, None,
            'unavailable', ref, 'evidence_port_unavailable'),), missing=('evidence_port',),
            provenance=tuple(r.ref for r in (before, after) if r), evidence_ref=ref), budget_bytes)


class EvidenceFacade:
    version = 'sluice-profile-evidence-v2'
    available = True

    @staticmethod
    def capabilities():
        from .profile_questions import topic_index
        return topic_index()

    def __init__(self, context, db=None):
        from .profile_adapter import ProfileAdapter
        self.adapter = ProfileAdapter(context, db)
        self.db = self.adapter.db
        self.binding = None

    def bind(self, hypothesis, attempt):
        self.binding = (hypothesis, attempt)

    def validate_question(self, question, mode):
        spec = QUESTIONS.get(question)
        if spec is None:
            raise ValueError('question_unsupported')
        return mode in spec.required_modes

    def register_evaluate_action(self, observation):
        return self.adapter.register_evaluate_action(observation).to_public_ref()

    def register_profile_action(self, observation):
        return self.adapter.register_profile_action(observation).to_public_ref()

    def observe(self, *, question, after, budget_bytes=3800):
        """Project baseline evidence through the same gates, without loop mutation.

        The private query binding supplies identity/signal checks only. The
        observation envelope deliberately carries no registered hypothesis ref.
        """
        from .contracts import Hypothesis
        if question not in QUESTIONS:
            raise ValueError('question_unsupported')
        spec = QUESTIONS[question]
        binding = self.binding
        try:
            self.bind(Hypothesis('observation-' + after.ref, self.adapter.context.case_id,
                after.source_hash, 'baseline_observation', (), (), spec.expected_signal_keys,
                (question,), 'observation_only', ()), {'child_source_hash': after.source_hash})
            bundle = self.query(question=question, before=None, after=after, budget_bytes=budget_bytes)
        finally:
            self.binding = binding
        value = bundle.to_dict()
        value.pop('hypothesis_ref')
        value['interpretation'] = 'baseline_observation'
        return value

    def member_refs(self, ref):
        handle = self.adapter.handles.get(ref.ref)
        if not handle:
            return ()
        def descend(h):
            return [item for r in h.member_refs for item in
                    [self.adapter.handles[r], *descend(self.adapter.handles[r])]]
        return tuple(h.to_public_ref() for h in descend(handle))

    def details(self, *, profile_ref, view, cursor=None, budget_bytes=4096, task_ref=None):
        """Read-only drill-down; opaque cursors never select arbitrary SQL or files."""
        from .evidence_guard import scan, require_ref
        from .profile_adapter import read_json
        from .controller import save
        require_ref(profile_ref, 'profile')
        if task_ref is not None:
            require_ref(task_ref,'evidence')
            if view != 'boundaries':
                raise ValueError('detail_task_filter_unsupported')
        handle = self.adapter.handles.get(profile_ref)
        if handle is None or handle.availability != 'available':
            raise ValueError('profile_reference_unknown')
        if view == 'device_identity':
            from .device_identity import details
            return details(self.adapter, handle, cursor, budget_bytes)
        if view == 'execution_work':
            return self._execution_work_details(handle, cursor, budget_bytes)
        if handle.member_refs and view == 'members':
            return self._member_details(handle, cursor, budget_bytes)
        if handle.member_refs:
            raise ValueError('profile_detail_requires_member_ref')
        allowed = {'runtime':{'tasks','critical_path','engine_occupancy','dispatch_delays','scheduler_phases','boundaries','host'}, 'pmu':{'pmu','pmu_samples'},
                   'compiler':{'resources','allocations','hints','pipeline'}, 'clean':{'samples','host','process_rounds','measured_host'}}
        if view not in allowed.get(handle.identity.profile_mode,set()):
            raise ValueError('detail_view_unsupported')
        if type(budget_bytes) is not int or not 1024 <= budget_bytes <= 4096:
            raise ValueError('evidence_budget_invalid')
        gate = self.db.compatibility(handle.db_run_id,handle.db_run_id)
        # Missing totals prevent PMU aggregate comparison, but valid sparse
        # counter observations can still be inspected without inventing totals.
        sparse_pmu = (view in ('pmu','pmu_samples') and handle.identity.profile_mode == 'pmu'
                      and set(gate['reasons']) <= {'pmu_sample_count_mismatch', 'pmu_dispatch_coverage_incomplete'})
        if not gate['compatible'] and not sparse_pmu:
            raise ValueError('capture_not_current')
        from .profile_batches import validate_original
        validate_original(self.db, handle)
        if view == 'pmu_samples':
            from ..profiling.pmu_samples import details
            return details(self.adapter, handle, cursor, budget_bytes)
        offset = 0
        detail_version = 'details-v1'
        if view in ('engine_occupancy', 'dispatch_delays'):
            detail_version = 'runtime-intervals-v1'
        if view == 'scheduler_phases':
            from profile_db.scheduler_observations import VERSION
            detail_version = VERSION
        if view in ('boundaries','tasks','critical_path'):
            from .task_boundaries import VERSION
            detail_version = VERSION+':'+view
        if view == 'process_rounds' or (view == 'host' and handle.identity.profile_mode == 'clean'):
            from .clean_processes import VERSION
            detail_version = VERSION+':'+view
        if view == 'host' and handle.identity.profile_mode == 'runtime':
            from .runtime_processes import VERSION
            detail_version = VERSION+':host'
        if view == 'measured_host':
            from .measured_host import VERSION
            detail_version = VERSION
        if view == 'hints':
            detail_version = 'native-hint-fragments-v2'
        binding = opaque('evidence',encoded([profile_ref,view,handle.artifact_fingerprint,detail_version]).decode())
        if task_ref is not None:
            binding = opaque('evidence',(binding,task_ref))
        cursor_root = self.adapter.context.work_root/'profile_cursors'
        if cursor is not None:
            require_ref(cursor,'evidence')
            document = read_json(cursor_root/(cursor.split(':')[1]+'.json'))
            if (document.get('binding') != binding or type(document.get('offset')) is not int
                    or document['offset'] < 0):
                raise ValueError('detail_cursor_mismatch')
            offset = document['offset']
        process_doc = None
        boundary_doc = None
        if view in ('process_rounds','boundaries','measured_host'):
            from types import SimpleNamespace
            if view == 'boundaries':
                from .task_boundaries import analyze
                boundary_doc = analyze(self.adapter, handle)
                # Preserve the original edge ordinal when selecting a task;
                # filtered offsets must not give different edges one identity.
                rows = [{'entity':i,**row} for i,row in enumerate(boundary_doc['rows'])]
                if task_ref is not None:
                    from .task_boundaries import task_ref as task_reference
                    known = {task_reference(profile_ref,r[0]) for r in self.db.connection.execute(
                        'SELECT task_id FROM task WHERE run_id=?',[handle.db_run_id]).fetchall()}
                    if task_ref not in known:
                        raise ValueError('boundary_task_reference_unknown')
                    rows = [r for r in rows if task_ref in (r['producer_task_ref'],r['consumer_task_ref'])]
            elif view=='measured_host':
                from .measured_host import analyze
                process_doc=analyze(self.adapter,handle)
                rows=process_doc['rows']
            else:
                from .clean_processes import analyze
                process_doc = analyze(self.adapter, handle)
                rows = process_doc['rows']
            # Feed the same bounded page builder, exposing only the allowlisted
            # measured fields below and capture-local process references.
            result = SimpleNamespace(truncated=False, facts=[
                SimpleNamespace(rec='DETAIL', fields={'entity':offset+i, **fields},
                                evidence=SimpleNamespace(value='measured' if view=='measured_host' else 'proven'))
                for i, fields in enumerate(rows[offset:offset+8])])
            result.facts.append(SimpleNamespace(rec='PAGE', fields={'has_more':offset+8 < len(rows)}))
        else:
            result = self.db.query('profile_details',run_id=handle.db_run_id,view=view,
                                   offset=offset,limit=8,budget_bytes=32768)
        host_process_doc,host_processes,host_mapping_reason = None,{},None
        if view == 'host' and handle.identity.profile_mode == 'clean':
            from .clean_processes import analyze
            from profile_db.errors import QueryError
            try:
                host_process_doc = analyze(self.adapter,handle)
                host_processes = {r['process_ref']:r for r in host_process_doc['rows']}
            except (ValueError,OSError,QueryError) as exc:
                # Host invocation costs can remain useful when measured-round
                # process reconciliation or optional device mapping is absent.
                reason = str(exc)
                host_mapping_reason = (reason if reason.startswith('clean_') and
                    all(c.islower() or c.isdigit() or c=='_' for c in reason)
                    else 'clean_process_mapping_unavailable')
        runtime_doc = self._runtime_process_mapping(handle) if view=='host' and handle.identity.profile_mode=='runtime' else None
        if result.truncated:
            raise ValueError('detail_page_incomplete')
        numeric = {'wall_us','busy_us','physical_rows','critical','position','compute_us','gap_us','stall_us',
                   'row_index','dispatch_us','receive_us','finish_us','dispatch_to_receive_us',
                   'receive_to_start_us','end_to_finish_us','window_start_us','window_end_us',
                   'tail_start_us','occupancy','tail_occupancy',
                   'block_num',
                   'block_ordinal','task_ordinal','sample','function','core','thread','cycles','total_cycles','usage_bytes','limit_bytes',
                   'headroom_bytes','largest_buffer_bytes','allocation_count','size_bytes','offset_bytes',
                   'line','column','hint_ordinal','fragment_index','fragment_count','redacted',
                   'effective_us','complete','inclusive_seconds','self_seconds','child_count',
                   'consistent','host_us','device_us','orch_us','sched_us',
                   'round','dispatch_count','round_max_fraction', 'scheduler_lane', 'ordinal',
                   'start_us', 'end_us', 'duration_us', 'loop_iter', 'tasks_processed', 'pop_hit', 'pop_miss'}
        text = {'name','engine','space','counter','core_type','event_type','hint_code','basis','self_basis','kernel','lane','pipe','phase','mode','service_boundary'}
        items = []
        for fact in result.facts:
            if fact.rec != 'DETAIL' or 'entity' not in fact.fields:
                continue
            fields = {}
            for key,value in fact.fields.items():
                if key in numeric and (value is None or type(value) in (int,float,bool)):
                    fields[key] = value
                elif key in text and isinstance(value,str):
                    try:
                        scan(value)
                        fields[key] = value
                    except ValueError:
                        pass
            if view == 'host':
                # Invocation entities differ on every row. Keep a capture-local
                # process reference so the agent can follow one process over
                # repeated calls without exposing host PIDs or inventing ranks.
                if type(fact.fields.get('pid')) is int:
                    fields['process_ref'] = opaque('evidence', (profile_ref, 'host_process', fact.fields['pid']))
                    if handle.identity.profile_mode == 'clean':
                        process = host_processes.get(fields['process_ref'],{})
                        fields.update(rank=process.get('rank'),device=process.get('device'))
                    elif runtime_doc is not None:
                        process = runtime_doc['mapping'].get(fact.fields['pid'],{})
                        fields.update(rank=process.get('rank'),device=process.get('device'),
                                      capture_phase=process.get('capture_phase'))
                if type(fact.fields.get('inv')) is int:
                    fields['invocation_ordinal'] = fact.fields['inv']
            if view == 'process_rounds':
                fields['process_ref'] = fact.fields['process_ref']
                fields.update(rank=fact.fields.get('rank'),device=fact.fields.get('device'))
            if view=='measured_host':
                fields={k:v for k,v in fact.fields.items() if k!='entity'}
            if view == 'hints':
                for key in ('message', 'source_file'):
                    value = fact.fields.get(key)
                    if value is not None:
                        scan({key: value})
                    fields[key] = value
                fields['location_kind'] = 'native_reported'
                fields['source_ref'] = (opaque('evidence', (profile_ref, fact.fields['source_identity']))
                    if fact.fields.get('source_identity') else None)
            if view == 'boundaries':
                fields = {k:v for k,v in fact.fields.items() if k != 'entity'}
            if view in ('tasks','critical_path','dispatch_delays'):
                from .task_boundaries import task_ref as task_reference
                fields['task_ref'] = task_reference(profile_ref, fact.fields['entity'])
            entity = (fact.fields.get('entity'),fact.fields.get('sample'),fact.fields.get('core'),fact.fields.get('thread'))
            if view == 'dispatch_delays':
                entity += (fact.fields.get('row_index'), fact.fields.get('engine'))
            items.append({'entity_ref':opaque('evidence',(profile_ref,view,entity)),
                          'status':{'proven':'derived','measured':'measured'}.get(fact.evidence.value,'unavailable'),'fields':fields})
        page = next((f.fields for f in result.facts if f.rec=='PAGE'),{})
        has_more = page.get('has_more',False)
        # Persist only private offsets; public cursors bind capture, view and
        # parser version and survive a controller restart.
        while True:
            following = offset+len(items)
            next_cursor = opaque('evidence',(binding,following)) if has_more else None
            value = {'profile_ref':profile_ref,'view':view,'items':items,'next_cursor':next_cursor,
                     'truncated':bool(next_cursor),'evidence_ref':binding}
            if view in ('scheduler_phases','engine_occupancy','dispatch_delays'):
                value.update(next(f.fields for f in result.facts if f.rec == 'SCOPE'))
            if process_doc is not None:
                keys=({'aggregation','warmup_excluded','origin','union_basis',
                       'end_to_end','parser_version','interpretation'}
                      if view=='measured_host' else set(process_doc)-{'rows','metrics'})
                value.update({k:v for k,v in process_doc.items() if k in keys})
                if view=='process_rounds':
                    value['interpretation'] = 'measured_round_process_dispatch_sums'
            if boundary_doc is not None:
                value.update({k:v for k,v in boundary_doc.items() if k not in ('rows','metrics')})
            if view == 'host':
                from profile_db.runtime_adapter.storage import read_snapshot, resolve
                host = read_snapshot(self.db.connection, handle.db_run_id, 'host')
                payload = host.get('payload', {})
                coverage = ('unavailable' if host.get('status') != 'ok' else
                            'incomplete' if payload.get('incomplete_records') != 0
                            or any(d.get('total', 0) > 0 for d in payload.get('drops', {}).values())
                            else 'complete')
                for (kind,) in self.db.connection.execute(
                        "SELECT kind FROM artifact WHERE run_id=? AND (kind LIKE 'host_log:%' OR kind='runtime_host_manifest')",
                        [handle.db_run_id]).fetchall():
                    resolve(self.db.connection, handle.db_run_id, kind)
                value.update(interpretation=('recorded_invocation_costs_including_warmup'
                    if handle.identity.profile_mode=='clean' else 'recorded_diagnostic_worker_invocation_costs'),
                             host_capture_status=coverage,
                             cold_compile_certified=False, steady_state_certified=False)
                if handle.identity.rank_label != 'single':
                    value.update(host_scope='fixture_execution', host_rank_mapping='unavailable')
                if handle.identity.profile_mode == 'clean':
                    verified = bool(host_process_doc and host_process_doc['rank_mapping_verified'])
                    value.update(host_scope='fixture_execution',
                        host_rank_mapping='measured_device_processes' if verified else 'unavailable',
                        rank_mapping_verified=verified,
                        rank_mapping_basis=host_process_doc['rank_mapping_basis'] if host_process_doc else None,
                        rank_mapping_reason=host_process_doc['rank_mapping_reason'] if host_process_doc else host_mapping_reason,
                        rank_mapping_scope='measured_processes_local_runtime_device_order')
                if runtime_doc is not None:
                    value.update({k:v for k,v in runtime_doc.items() if k!='mapping'})
                    value.update(host_scope='fixture_execution',
                        host_rank_mapping='diagnostic_device_processes' if runtime_doc['rank_mapping_verified'] else 'unavailable')
            scan(value)
            if len(encoded(value)) <= budget_bytes:
                if next_cursor:
                    save(cursor_root/(next_cursor.split(':')[1]+'.json'),{'binding':binding,'offset':following})
                return value
            if len(items) <= 1:
                raise ValueError('evidence_budget_invalid')
            items.pop()
            has_more = True

    def _runtime_process_mapping(self, handle):
        """Optional identity is independent of valid runtime tasks/host costs."""
        from .runtime_processes import analyze, VERSION
        from profile_db.errors import QueryError
        try:
            return analyze(self.adapter,handle)
        except (ValueError,OSError,QueryError,KeyError,TypeError) as exc:
            reason = str(exc)
            if not (reason.startswith('runtime_process_mapping_')
                    and all(c.islower() or c.isdigit() or c=='_' for c in reason)):
                reason = 'runtime_process_mapping_unavailable'
            return {'mapping':{},'rank_mapping_verified':False,'rank_mapping_basis':None,
                    'rank_mapping_reason':reason,
                    'rank_mapping_scope':'diagnostic_device_processes_local_runtime_device_order',
                    'parser_version':VERSION}

    def _execution_work_details(self, handle, cursor, budget_bytes):
        from .controller import save
        from .evidence_guard import require_ref, scan
        from .profile_adapter import read_json
        from ..profiling.execution_work import VERSION, analyze
        if handle.identity.profile_mode != 'runtime':
            raise ValueError('detail_view_unsupported')
        if type(budget_bytes) is not int or not 1024 <= budget_bytes <= 4096:
            raise ValueError('evidence_budget_invalid')
        fixtures = ([self.adapter.handles[r] for r in handle.member_refs]
                    if handle.member_kind == 'fixture' else [handle])
        binding = opaque('evidence', encoded([handle.ref, handle.artifact_fingerprint, VERSION]).decode())
        root = self.adapter.context.work_root / 'profile_cursors'
        offset = 0
        if cursor:
            require_ref(cursor, 'evidence')
            doc = read_json(root / (cursor.split(':')[1] + '.json'))
            if (doc.get('binding') != binding or type(doc.get('offset')) is not int
                    or not 0 <= doc['offset'] < len(fixtures)):
                raise ValueError('detail_cursor_mismatch')
            offset = doc['offset']
        items = []
        for fixture in fixtures[offset:offset + 8]:
            doc = analyze(self.adapter, fixture)
            items.append({'entity_ref': opaque('evidence', (binding, fixture.ref)), 'status': 'derived',
                          'fields': {'fixture_id': fixture.identity.fixture_id, 'profile_ref': fixture.ref,
                                     **doc['metrics'], 'scope': doc['scope'],
                                     'complete_registered_members': doc['complete_registered_members']}})
        while True:
            following = offset + len(items)
            more = following < len(fixtures)
            next_cursor = opaque('evidence', (binding, following)) if more else None
            value = {'profile_ref': handle.ref, 'view': 'execution_work', 'items': items,
                     'fixture_count': len(fixtures), 'next_cursor': next_cursor, 'truncated': more,
                     'evidence_ref': binding, 'dispatch_count_unit': 'rank_dispatch_capture',
                     'time_aggregation': False, 'performance_claim': False}
            scan(value)
            if len(encoded(value)) <= budget_bytes:
                if next_cursor:
                    save(root / (next_cursor.split(':')[1] + '.json'), {'binding': binding, 'offset': following})
                return value
            if len(items) <= 1:
                raise ValueError('evidence_budget_invalid')
            items.pop()

    def _execution_work_query(self, spec, bundle, before, after, budget_bytes):
        from ..profiling.execution_work import compare
        try:
            result = compare(self.adapter, before, after, source_pair=(
                self.binding[0].parent_source_hash, self.binding[1]['child_source_hash']))
        except Exception:
            return bounded(replace(bundle, missing=('execution_work_incompatible',), comparison='incomparable'),
                           budget_bytes)
        values = [('comparison_scope', result['comparison_scope'], None)]
        changes = []
        for signal in spec.signals:
            metric = 'logical_tasks' if signal.metric == 'tasks' else signal.metric
            values.extend(((signal.key, result['after'][metric], signal.unit),
                           (signal.key + '_delta', result['deltas'][metric], signal.unit)))
            changes.append(result['deltas'][metric])
        values.extend((('dispatch_count', result['after']['dispatch_count'], 'count'),
                       ('dispatch_count_delta', result['deltas']['dispatch_count'], 'count'),
                       ('rank_count', result['after']['rank_count'], 'count'),
                       ('dispatch_count_unit', 'rank_dispatch_capture', None),
                       ('time_aggregation', False, None), ('performance_claim', False, None)))
        changes.append(result['deltas']['dispatch_count'])
        comparison = ('no_change' if not any(changes) else 'decreased' if all(d <= 0 for d in changes)
                      else 'increased' if all(d >= 0 for d in changes) else 'changed')
        facts = tuple(EvidenceFact(spec.question_id, k, v, unit, 'derived', bundle.evidence_ref, None)
                      for k, v, unit in values)
        return bounded(replace(bundle, facts=facts, comparison=comparison), budget_bytes, 12)

    def _member_details(self, handle, cursor, budget_bytes):
        from .controller import save
        from .evidence_guard import require_ref, scan
        from .profile_adapter import read_json
        if type(budget_bytes) is not int or not 1024 <= budget_bytes <= 4096:
            raise ValueError('evidence_budget_invalid')
        binding = opaque('evidence', (handle.ref, handle.member_refs, handle.member_slots, 'member-pages-v1'))
        root = self.adapter.context.work_root/'profile_cursors'
        offset = 0
        if cursor:
            require_ref(cursor,'evidence')
            doc = read_json(root/(cursor.split(':')[1]+'.json'))
            if (doc.get('binding') != binding or type(doc.get('offset')) is not int
                    or not 0 <= doc['offset'] < len(handle.member_refs)):
                raise ValueError('detail_cursor_mismatch')
            offset = doc['offset']
        items = []
        for index,ref in enumerate(handle.member_refs[offset:offset+8],offset):
            child = self.adapter.handles[ref]
            fields = {'profile_ref':ref,'mode':child.identity.profile_mode,
                      'slot':handle.member_slots[index] if handle.member_slots else child.identity.rank_label}
            if child.member_kind:
                fields['member_kind'] = child.member_kind
            if not child.member_refs and child.identity.rank_label != 'single':
                fields.update(rank=child.identity.device['rank'], device=child.identity.device['device'],
                              dispatch=child.identity.invocation)
            items.append({'entity_ref':opaque('evidence',(handle.ref,'member',index)),
                          'status':'derived','fields':fields})
        while True:
            following = offset+len(items)
            more = following < len(handle.member_refs)
            next_cursor = opaque('evidence',(binding,following)) if more else None
            value = {'profile_ref':handle.ref,'view':'members','items':items,'member_count':len(handle.member_refs),
                     'next_cursor':next_cursor,'truncated':more,'evidence_ref':binding}
            scan(value)
            if len(encoded(value)) <= budget_bytes:
                if next_cursor:
                    save(root/(next_cursor.split(':')[1]+'.json'),{'binding':binding,'offset':following})
                return value
            if len(items) <= 1:
                raise ValueError('evidence_budget_invalid')
            items.pop()


    def _fixture_query(self, spec, bundle, handles, before, after, budget_bytes):
        """A batch is a set of scenarios, never an invented common timeline."""
        ref, question = bundle.evidence_ref, spec.question_id
        if any(h.member_kind != 'fixture' or not h.member_refs or h.availability != 'available' for h in handles):
            return bounded(replace(bundle, missing=('fixture_batch_mismatch',), comparison='incomparable'), budget_bytes)
        arms = [{self.adapter.handles[r].identity.fixture_id:self.adapter.handles[r]
                 for r in h.member_refs} for h in handles]
        if any(set(arm) != set(arms[-1]) for arm in arms):
            return bounded(replace(bundle, missing=('fixture_batch_mismatch',), comparison='incomparable'), budget_bytes)
        summary = [('fixture_count',len(arms[-1]),'count'), ('end_to_end',False,None)]
        comparison, missing = 'unavailable', set()
        projected = []
        for slot, child in sorted(arms[-1].items()):
            old = arms[0].get(slot) if before else None
            item = self.query(question=question, before=(old.to_public_ref() if old else None), after=child.to_public_ref(),
                budget_bytes=budget_bytes)
            projected.append(item)
            missing.update(item.missing)
        if any(p.status != 'available' for p in projected):
            return bounded(replace(bundle, missing=tuple(sorted(missing | {'fixture_evidence_incomplete'})),
                           comparison='incomparable' if any(p.comparison == 'incomparable' for p in projected) else 'unavailable'),
                           budget_bytes)
        if before:
            decisions = {p.comparison for p in projected}
            comparison = ('incomparable' if 'incomparable' in decisions else
                          'unavailable' if 'unavailable' in decisions else
                          next(iter(decisions)) if len(decisions) == 1 else 'changed')
        facts = tuple(EvidenceFact(question,k,v,unit,'derived',ref,None) for k,v,unit in summary)
        facts += tuple(replace(f,key=slot+'_'+f.key,source=ref)
                       for slot,item in zip(sorted(arms[-1]),projected) for f in item.facts)
        return bounded(replace(bundle, facts=facts, comparison=comparison,
            missing=tuple(sorted(missing)), truncated=any(p.truncated for p in projected)), budget_bytes,64)

    def query(self, *, question, before, after, budget_bytes=4096):
        if self.binding is None:
            raise ValueError('question_not_bound')
        hypothesis, attempt = self.binding
        if question not in hypothesis.verification_questions or question not in QUESTIONS:
            raise ValueError('question_not_registered')
        spec = QUESTIONS[question]
        ref = opaque('evidence', (hypothesis.hypothesis_id, question,
                                before.ref if before else None, after.ref if after else None))
        bundle = EvidenceBundle(question, evidence_ref=ref, mechanism_id=hypothesis.mechanism,
            hypothesis_ref=opaque('hypothesis', hypothesis.hypothesis_id),
            program_ref=opaque('program', attempt['child_source_hash']),
            provenance=tuple(r.ref for r in (before, after) if r))

        def unavailable(reason, comparison='unavailable', reasons=()):
            return bounded(replace(bundle, missing=tuple(sorted({reason, *reasons})), comparison=comparison), budget_bytes)

        if not set(spec.expected_signal_keys) <= set(hypothesis.expected_signals):
            return unavailable('signal_unregistered')
        if after is None:
            return unavailable('capture_missing')
        handles = [self.adapter.handles.get(r.ref) for r in (before, after) if r]
        if any(h is None for h in handles):
            return unavailable('profile_reference_unknown')
        ha = handles[-1]
        if ha.availability != 'available':
            return unavailable(ha.reason or 'capture_missing')
        if ha.identity.profile_mode not in spec.required_modes:
            return unavailable('mode_unsupported')
        if (ha.identity.case_id != hypothesis.case_id or ha.identity.source_digest != attempt['child_source_hash']):
            code = 'case_id_mismatch' if ha.identity.case_id != hypothesis.case_id else 'source_mismatch'
            return unavailable('identity_mismatch', 'incomparable', reasons=(code,))
        if before is not None:
            hb = handles[0]
            if hb.availability != 'available':
                return unavailable(hb.reason or 'capture_missing')
            if hb.identity.source_digest != hypothesis.parent_source_hash:
                return unavailable('identity_mismatch', 'incomparable', reasons=('source_mismatch',))
        if ha.member_kind == 'fixture' or before is not None and hb.member_kind == 'fixture':
            return self._fixture_query(spec, bundle, handles, before, after, budget_bytes)
        if (before is not None and question in ('did_task_granularity_change?', 'how_did_dispatched_work_change?')
                and (ha.member_refs or hb.member_refs)):
            def labels(handle):
                return {self.adapter.handles[r].identity.rank_label for r in handle.member_refs} if handle.member_refs else {handle.identity.rank_label}
            if labels(ha) != labels(hb):
                return self._execution_work_query(spec, bundle, hb, ha, budget_bytes)
        if ha.member_refs or before is not None and hb.member_refs:
            children = [self.adapter.handles[r] for r in ha.member_refs]
            old = {self.adapter.handles[r].identity.rank_label:self.adapter.handles[r]
                   for r in hb.member_refs} if before is not None else {}
            if not children or before is not None and set(old)!={c.identity.rank_label for c in children}:
                return unavailable('rank_membership_mismatch','incomparable')
            if question == 'how_large_is_recorded_host_cost?':
                # One fixture worker can drive several ranks/dispatches. Its
                # host stream is stored once; PID order does not identify ranks.
                child=children[0]
                prior=old.get(child.identity.rank_label)
                item=self.query(question=question,before=prior.to_public_ref() if prior else None,
                                after=child.to_public_ref(),budget_bytes=budget_bytes)
                if item.status != 'available':
                    return bounded(replace(bundle,missing=item.missing,comparison=item.comparison),budget_bytes)
                mapping = self._runtime_process_mapping(child)
                return bounded(replace(bundle,facts=(EvidenceFact(question,'host_scope','fixture_execution',None,
                    'derived',ref,None),EvidenceFact(question,'host_rank_mapping',
                    'diagnostic_device_processes' if mapping['rank_mapping_verified'] else None,None,
                    'derived' if mapping['rank_mapping_verified'] else 'unavailable',ref,
                    mapping['rank_mapping_reason']))+tuple(replace(f,source=ref) for f in item.facts),
                    missing=item.missing,comparison=item.comparison,truncated=item.truncated),budget_bytes,spec.max_facts)
            projected = []
            for child in children:
                prior = old.get(child.identity.rank_label)
                item = self.query(question=question,before=prior.to_public_ref() if prior else None,
                                  after=child.to_public_ref(),budget_bytes=budget_bytes)
                projected.append(item)
            facts = tuple(replace(f,key=child.identity.rank_label+'_'+f.key,source=ref)
                          for child,item in zip(children,projected) for f in item.facts)
            if question == 'which_rank_has_longest_local_span?':
                durations = [(child.identity.rank_label,f.value) for child,item in zip(children,projected)
                             for f in item.facts if f.key=='rank_local_span_us' and f.status!='unavailable']
                if len(durations)==len(children) and durations:
                    slowest = max(durations,key=lambda pair:pair[1])
                    fastest = min(v for _,v in durations)
                    summary = [('longest_member',slowest[0],None),('longest_local_span_us',slowest[1],'us'),
                               ('local_duration_spread_us',slowest[1]-fastest,'us')]
                    if fastest>0:
                        summary.append(('local_duration_ratio',slowest[1]/fastest,'fraction'))
                    facts = tuple(EvidenceFact(question,key,value,unit,'derived',ref,None)
                                  for key,value,unit in summary)+facts
            decisions = {item.comparison for item in projected}
            comparison = (next(iter(decisions)) if len(decisions)==1 else
                          'unavailable' if decisions & {'unavailable','incomparable'} else 'changed')
            return bounded(replace(bundle,facts=facts,
                missing=tuple(sorted({m for item in projected for m in item.missing})),comparison=comparison,
                truncated=any(item.truncated for item in projected)),budget_bytes,32)
        try:
            # The strict gate covers core topology and analysis semantics even
            # for a single run; legacy captures cannot invent missing identity.
            source_pair = (hypothesis.parent_source_hash, attempt['child_source_hash'])
            gate = self.db.compatibility(handles[0].db_run_id, ha.db_run_id, source_pair=source_pair)
            coverage_question = question == 'is_pmu_dispatch_coverage_complete?'
            coverage_only = coverage_question and set(gate['reasons']) <= {
                'pmu_dispatch_coverage_incomplete', 'pmu_sample_count_mismatch'}
            if not gate['compatible'] and not coverage_only:
                reason = ('capture_repair_required' if 'capture_repair_required' in gate['reasons'] else
                          'analysis_stale' if 'analysis_stale' in gate['reasons'] else 'identity_mismatch')
                from .comparison_diagnostics import reasons as comparison_reasons
                codes = comparison_reasons(gate['reasons'],
                    {'device': handles[0].identity.device}, {'device': ha.identity.device})
                return unavailable(reason, 'incomparable' if before else 'unavailable', reasons=codes)
            from .profile_batches import validate_original
            for handle in handles:
                validate_original(self.db, handle)
            if spec.query_plan == ('boundary_summary',):
                from .task_boundaries import analyze
                try:
                    docs = [analyze(self.adapter, h) for h in handles]
                except ValueError as exc:
                    reason = str(exc)
                    return unavailable(reason if reason.startswith('boundary_') else 'boundary_evidence_invalid')
                facts = [EvidenceFact(question,s.key,docs[-1]['metrics'][s.metric],s.unit,
                                      'derived',ref,None) for s in spec.signals]
                comparison = 'unavailable'
                if before is not None:
                    deltas = [docs[-1]['metrics'][s.metric]-docs[0]['metrics'][s.metric] for s in spec.signals]
                    facts.extend(EvidenceFact(question,s.key+'_delta',d,s.unit,'derived',ref,None)
                                 for s,d in zip(spec.signals,deltas))
                    comparison = 'changed' if any(deltas) else 'no_change'
                for key in ('interpretation','measured_copy_traffic','copy_requirement_verified','whole_chain_speed_verified'):
                    facts.append(EvidenceFact(question,key,docs[-1][key],None,'derived',ref,None))
                facts.append(EvidenceFact(question,'comparison_scope','boundary_counts_only',None,'derived',ref,None))
                return bounded(replace(bundle,facts=tuple(facts),comparison=comparison),
                               min(budget_bytes,spec.default_budget_bytes),spec.max_facts)
            if spec.query_plan in (('clean_process_summary',),('measured_host_summary',)):
                measured_host=spec.query_plan==('measured_host_summary',)
                if measured_host:
                    from .measured_host import analyze
                else:
                    from .clean_processes import analyze
                try:
                    docs = [analyze(self.adapter, h) for h in handles]
                except ValueError as exc:
                    reason = str(exc)
                    return unavailable(reason if reason.startswith('clean_') else 'clean_process_evidence_invalid')
                facts = [EvidenceFact(question, s.key, docs[-1]['metrics'][s.metric], s.unit,
                                      'derived', ref, None) for s in spec.signals]
                comparison = 'unavailable'
                if before is not None:
                    changes = []
                    for signal in spec.signals:
                        delta = docs[-1]['metrics'][signal.metric] - docs[0]['metrics'][signal.metric]
                        facts.append(EvidenceFact(question, signal.key+'_delta', delta, signal.unit,
                                                  'derived', ref, None))
                        changes.append(delta)
                    comparison = 'changed' if any(changes) else 'no_change'
                metadata=(('aggregation','warmup_excluded','origin','end_to_end')
                          if measured_host else ('aggregation','warmup_excluded','rank_mapping_verified',
                                                 'communication_causality_verified'))
                for key in metadata:
                    facts.append(EvidenceFact(question, key, docs[-1][key], None, 'derived', ref, None))
                if not measured_host:
                    verified = docs[-1]['rank_mapping_verified']
                    facts.append(EvidenceFact(question,'rank_mapping_basis',docs[-1]['rank_mapping_basis'],None,
                        'derived' if verified else 'unavailable',ref,docs[-1]['rank_mapping_reason']))
                return bounded(replace(bundle, facts=tuple(facts), comparison=comparison),
                               min(budget_bytes, spec.default_budget_bytes), spec.max_facts)
            if spec.query_plan == ('host_cost_summary',):
                from profile_db.runtime_adapter.storage import read_snapshot, resolve
                for handle in handles:
                    doc = read_snapshot(self.db.connection, handle.db_run_id, 'host')
                    payload = doc.get('payload', {})
                    if (doc.get('status') != 'ok' or payload.get('incomplete_records') != 0
                            or any(d.get('total', 0) > 0 for d in payload.get('drops', {}).values())):
                        return unavailable('host_capture_incomplete')
                    for (kind,) in self.db.connection.execute(
                            "SELECT kind FROM artifact WHERE run_id=? AND (kind LIKE 'host_log:%' OR kind='runtime_host_manifest')",
                            [handle.db_run_id]).fetchall():
                        resolve(self.db.connection, handle.db_run_id, kind)
            result = self.db.query(spec.query_plan[0], run_id=ha.db_run_id, budget_bytes=4096)
            polling_scope = None
            if spec.query_plan == ('async_poll_summary',):
                polling_scope = next((f.fields for f in result.facts if f.rec == 'SCOPE'), {})
                if polling_scope.get('reason') == 'scheduler_phase_capture_absent':
                    return unavailable('scheduler_phase_capture_absent')
            facts, missing = [], []
            for signal in spec.signals:
                matched = next((f for f in result.facts
                              if f.rec == 'METRIC' and f.evidence.value in ('measured', 'proven')
                              and signal.metric in f.fields), None)
                value = matched.fields[signal.metric] if matched is not None else None
                valid = type(value) in (int, float) and math.isfinite(value)
                facts.append(EvidenceFact(question, signal.key, value if valid else None, signal.unit,
                    ('measured' if matched.evidence.value == 'measured' else 'derived') if valid else 'unavailable',
                    ref, None if valid else 'no_matching_fact'))
                if not valid:
                    missing.append('no_matching_fact')
            comparison = 'unavailable'
            if spec.query_plan == ('measurement_summary',):
                count = next((f.fields['count'] for f in result.facts if f.rec == 'METRIC'
                              and f.evidence.value in ('measured', 'proven') and 'count' in f.fields), None)
                if type(count) is int and count > 0:
                    facts.append(EvidenceFact(question, 'sample_count', count, 'count', 'derived', ref, None))
                if before is not None and not missing:
                    decision = self.db.compare_benchmarks(handles[0].db_run_id, ha.db_run_id, source_pair=source_pair)
                    comparison = decision['comparison']
                    missing.extend(decision.get('reasons', []))
                    stats = [('before_median_us', decision.get('before', {}).get('median_us'), 'us'),
                             ('delta_us', decision.get('delta_us'), 'us'),
                             ('relative_change', decision.get('relative_change'), 'fraction'),
                             ('before_sample_count', decision.get('before', {}).get('count'), 'count'),
                             ('after_sample_count', decision.get('after', {}).get('count'), 'count')]
                    facts.append(EvidenceFact(question, 'interpretation', 'observed_sample_medians', None,
                                              'derived', ref, None))
                    for key, value, unit in stats:
                        if type(value) in (int, float) and math.isfinite(value):
                            facts.append(EvidenceFact(question, key, value, unit, 'derived', ref, None))
                return bounded(replace(bundle, facts=tuple(facts), missing=tuple(missing), comparison=comparison,
                                       truncated=result.truncated), min(budget_bytes, spec.default_budget_bytes), spec.max_facts)
            if before is not None and (not missing or spec.query_plan == ('host_cost_summary',)):
                comparison_result = (self.db.compare(handles[0].db_run_id, ha.db_run_id, source_pair=source_pair)
                                     if ha.identity.profile_mode != 'clean' and not coverage_question else None)
                if comparison_result is not None and not any(
                        f.rec == 'COMPARE' and f.fields.get('profile_compatible') is True
                        for f in comparison_result.facts):
                    return unavailable('identity_mismatch', 'incomparable')
                changes = []
                before_metrics = None
                for signal in spec.signals:
                    delta = next((f.fields.get('delta') for f in comparison_result.facts
                                  if f.rec == 'DELTA' and f.fields.get('metric') == signal.metric), None) if comparison_result else None
                    if delta is None:
                        if before_metrics is None:
                            prior = self.db.query(spec.query_plan[0], run_id=handles[0].db_run_id, budget_bytes=4096)
                            if polling_scope is not None and any(
                                    f.rec == 'SCOPE' and f.fields.get('reason') == 'scheduler_phase_capture_absent'
                                    for f in prior.facts):
                                return unavailable('before_scheduler_phase_capture_absent')
                            before_metrics = {k:v for f in prior.facts if f.rec == 'METRIC'
                                              and f.evidence.value in ('measured','proven') for k,v in f.fields.items()}
                        prior_value = before_metrics.get(signal.metric)
                        current = next((f.value for f in facts if f.key == signal.key), None)
                        if all(type(v) in (int,float) and math.isfinite(v) for v in (prior_value,current)):
                            delta = current-prior_value
                    if type(delta) not in (int, float) or not math.isfinite(delta):
                        missing.append('before_no_matching_fact')
                        continue
                    facts.append(EvidenceFact(question, signal.key + '_delta', delta, signal.unit, 'derived', ref, None))
                    # Diagnostic quantities describe mechanisms, not speed.
                    # Fusion can reduce tasks while increasing cycles or total
                    # latency; only clean samples can establish a speed verdict.
                    changes.append('no_change' if abs(delta) <= signal.tolerance else
                                   'decreased' if delta < 0 else 'increased')
                if missing:
                    # A single diagnostic invocation has no later-invocation
                    # statistic. Preserve comparable known deltas without an
                    # overall verdict over the missing host measurements.
                    scope = 'available_host_metrics_only' if spec.query_plan == ('host_cost_summary',) else 'available_metrics_only'
                    facts.append(EvidenceFact(question,'comparison_scope',scope,None,
                                              'derived',ref,None))
                elif changes and len(set(changes)) == 1:
                    comparison = changes[0]
                elif changes:
                    comparison = 'changed'
            if question == 'what_are_repeated_host_costs?':
                facts.append(EvidenceFact(question, 'measurement_scope',
                    'per_process_invocations_including_warmup', None, 'derived', ref, None))
                facts.append(EvidenceFact(question, 'cold_compile_certified', False, None, 'derived', ref, None))
            if polling_scope is not None:
                facts.extend(EvidenceFact(question, key, value, None, 'derived', ref, None)
                             for key, value in polling_scope.items())
            if ha.identity.profile_mode == 'pmu' and question != 'is_pmu_dispatch_coverage_complete?':
                coverage = dict(self.db.connection.execute(
                    "SELECT metric,value FROM diagnostic_metric WHERE run_id=? AND metric IN ('pmu_coverage_complete','pmu_missing_samples','pmu_extra_samples')",
                    [ha.db_run_id]).fetchall())
                facts.extend(EvidenceFact(question, key, value, 'flag' if key.endswith('complete') else 'count',
                                          'derived', ref, None) for key, value in sorted(coverage.items()))
            return bounded(replace(bundle, facts=tuple(facts), missing=tuple(missing), comparison=comparison,
                                   truncated=result.truncated), min(budget_bytes, spec.default_budget_bytes),
                           spec.max_facts + (3 if ha.identity.profile_mode == 'pmu' else 0))
        except ValueError as exc:
            return unavailable(str(exc) if str(exc) in ('diagnostic_original_invalid', 'capture_original_invalid')
                               else 'query_failed')
        except Exception:
            return unavailable('query_failed')


def configured_evidence(context, selection=None):
    """Explicit production injection; absent configuration stays fail closed."""
    if selection in (None, 'unavailable'):
        return UnavailableEvidencePort()
    if selection == 'profile_db':
        return EvidenceFacade(context)
    raise ValueError('evidence_port_unsupported')
