"""Controlled profiling question registry; query names and parameters are never agent input."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SignalSpec:
    key: str
    metric: str
    unit: str
    tolerance: float = 0


@dataclass(frozen=True)
class QuestionSpec:
    question_id: str
    owner_question: str
    required_modes: tuple[str, ...]
    query_plan: tuple[str, ...]
    signals: tuple[SignalSpec, ...]
    comparison_policy: str = 'optional_before_after'
    max_facts: int = 3
    default_budget_bytes: int = 4096

    @property
    def expected_signal_keys(self):
        return tuple(s.key for s in self.signals)


TASK_COUNT = QuestionSpec(
    'did_task_granularity_change?',
    'Recorded logical task count.',
    ('runtime',), ('overview',),
    (SignalSpec('task_count_down', 'tasks', 'count'),))
RUNTIME_SPAN = QuestionSpec(
    'did_profiled_span_change?', 'Did the profiled device span decrease?',
    ('runtime',), ('overview',),
    (SignalSpec('profiled_span_down', 'makespan_us', 'us'),))
PMU_CYCLES = QuestionSpec(
    'did_pmu_total_cycles_change?', 'Did total sampled PMU cycles decrease under the same event configuration?',
    ('pmu',), ('diagnostic_summary',),
    (SignalSpec('pmu_cycles_down', 'pmu_total_cycles', 'cycles'),))
COMPILER_HINTS = QuestionSpec(
    'did_compiler_hints_change?', 'Did the number of emitted compiler hints change?',
    ('compiler',), ('diagnostic_summary',),
    (SignalSpec('compiler_hints_changed', 'compiler_hint_count', 'count'),))
BUFFER_PEAK = QuestionSpec(
    'did_buffer_peak_change?', 'Did the largest per-kernel memory-space allocation footprint decrease?',
    ('compiler',), ('diagnostic_summary',),
    (SignalSpec('buffer_peak_down', 'buffer_peak_bytes', 'bytes'),))
PIPELINE_TIME = QuestionSpec(
    'did_pipeline_elapsed_change?', 'Did full pipeline elapsed time decrease (may include device execution)?',
    ('compiler',), ('diagnostic_summary',),
    (SignalSpec('pipeline_elapsed_down', 'pipeline_seconds', 's'),))
CLEAN_LATENCY = QuestionSpec(
    'did_clean_latency_change?', 'What are the native clean samples and the observed median difference?' ,
    ('clean',), ('measurement_summary',),
    (SignalSpec('clean_latency_down', 'median_us', 'us'),), max_facts=12)
PHYSICAL_WORK = QuestionSpec(
    'how_did_dispatched_work_change?', 'How many logical tasks and physical dispatch rows were observed?',
    ('runtime',), ('runtime_performance',),
    (SignalSpec('logical_tasks', 'logical_tasks', 'count'), SignalSpec('physical_rows', 'physical_rows', 'count')), max_facts=6)
CRITICAL_WORK = QuestionSpec(
    'where_is_critical_work?', 'How much observed critical-path time is compute versus inter-task gaps?',
    ('runtime',), ('runtime_performance',),
    tuple(SignalSpec(key, key, 'us') for key in ('cpm_us', 'observed_path_compute_us', 'observed_path_gap_us')), max_facts=8)
ENGINE_OCCUPANCY = QuestionSpec(
    'how_busy_are_compute_engines?', 'What fraction of available AIC and AIV core time is covered by observed work?',
    ('runtime',), ('runtime_performance',),
    tuple(SignalSpec(key, key, 'fraction') for key in ('aic_occupancy','aiv_occupancy','aic_tail_occupancy','aiv_tail_occupancy')), max_facts=10)
DISPATCH_DELAYS = QuestionSpec(
    'where_are_dispatch_delays?', 'What physical-row coordinate delays were observed? These sums are not exposed wall time or causal waits.',
    ('runtime',), ('runtime_performance',),
    tuple(SignalSpec(key, key, 'core_us') for key in ('dispatch_to_receive_core_us','receive_to_start_core_us','end_to_finish_core_us')), max_facts=8)
MEMORY_HEADROOM = QuestionSpec(
    'how_much_local_memory_headroom?', 'What is the minimum remaining compiler memory-space budget, and how large is one allocation?',
    ('compiler',), ('compiler_resource_summary',),
    (SignalSpec('minimum_headroom_bytes','minimum_headroom_bytes','bytes'),
     SignalSpec('maximum_space_utilization','maximum_space_utilization','fraction'),
     SignalSpec('largest_buffer_bytes','largest_buffer_bytes','bytes')), max_facts=8)
PMU_ACTIVITY = QuestionSpec(
    'which_pipelines_are_active?', 'What fraction of sampled cycles reports Scalar, Vec, Cube or MTE activity? Pipelines may overlap.',
    ('pmu',), ('pmu_activity',),
    tuple(SignalSpec(pipe+'_cycle_fraction',pipe+'_cycle_fraction','fraction')
          for pipe in ('scalar','vec','cube','mte1','mte2','mte3')), max_facts=14)
PMU_COVERAGE = QuestionSpec(
    'is_pmu_dispatch_coverage_complete?', 'Do native PMU task/core/thread identity counts match the independently recorded physical dispatch rows, including repetitions?',
    ('pmu',), ('diagnostic_summary',),
    (SignalSpec('pmu_coverage_fraction','pmu_coverage_fraction','fraction'),
     SignalSpec('pmu_coverage_complete','pmu_coverage_complete','flag')) +
    tuple(SignalSpec(key,key,'count') for key in ('pmu_expected_samples','pmu_observed_samples',
                                               'pmu_missing_samples','pmu_extra_samples')), max_facts=12)
RANK_BALANCE = QuestionSpec(
    'which_rank_has_longest_local_span?', 'Which rank has the longest independently measured local device duration? Clocks are not assumed synchronized.',
    ('runtime',), ('overview',), (SignalSpec('rank_local_span_us','makespan_us','us'),), max_facts=8)
HOST_COST = QuestionSpec(
    'how_large_is_recorded_host_cost?', 'What were first and later recorded host invocation costs? Host detail may verify local device identities and graph/timing lifetimes; first does not necessarily mean cold compile.',
    ('runtime',), ('host_cost_summary',),
    tuple(SignalSpec(key,key,'us') for key in ('first_host_us','later_host_us','first_device_us','later_device_us')),max_facts=10)
CLEAN_HOST_COST = QuestionSpec(
    'what_are_repeated_host_costs?',
    'What are first and later recorded invocation costs from the existing clean benchmark stream, including warmups? Per-process medians are not cold compile or end-to-end latency.',
    ('clean',), ('host_cost_summary',),
    tuple(SignalSpec(key, key, 'us') for key in
          ('first_host_us', 'later_host_us', 'first_device_us', 'later_device_us')) +
    tuple(SignalSpec(key, key, 'count') for key in ('first_invocations', 'later_invocations')),
    max_facts=15)
CLEAN_PROCESSES = QuestionSpec(
    'how_balanced_are_clean_processes?',
    'How different are per-process effective dispatch sums in measured clean rounds? Warmup excluded; exact clean samples reconciled. Validated startup records may identify runtime-visible devices and local rank order; dispatch sums do not prove communication causes.',
    ('clean',), ('clean_process_summary',),
    (SignalSpec('process_count', 'process_count', 'count'),
     SignalSpec('measured_rounds', 'measured_rounds', 'count'),
     SignalSpec('median_process_spread_us', 'median_process_spread_us', 'us'),
     SignalSpec('maximum_process_spread_us', 'maximum_process_spread_us', 'us'),
     SignalSpec('median_process_ratio', 'median_process_ratio', 'fraction')), max_facts=15)
TASK_BOUNDARIES = QuestionSpec(
    'what_crosses_task_boundaries?',
    'Which recorded tensor views cross producer and consumer tasks? View differences do not prove copies or performance changes.',
    ('runtime',), ('boundary_summary',),
    tuple(SignalSpec(key,key,'count') for key in
          ('dependency_edges','tensor_edges','identical_views','different_views','unavailable_views')), max_facts=15)
MEASURED_HOST = QuestionSpec(
    'what_are_measured_host_costs?',
    'Native host timings for measured clean rounds; warmup excluded. Read measured_host for round, process and dispatch values.',
    ('clean',), ('measured_host_summary',),
    (SignalSpec('measured_rounds','measured_rounds','count'),
     SignalSpec('median_host_us','median_host_us','us'),
     SignalSpec('maximum_host_us','maximum_host_us','us')),max_facts=12)
ASYNC_POLLING = QuestionSpec(
    'how_much_async_completion_polling_is_recorded?',
    'How much scheduler async-completion polling activity is recorded? Coverage is uncertified; lane unions do not establish DMA wait time or task/engine attribution.',
    ('runtime',), ('async_poll_summary',),
    (SignalSpec('recorded_async_poll_records', 'recorded_async_poll_records', 'count'),
     SignalSpec('recorded_async_poll_lanes', 'recorded_async_poll_lanes', 'count'),
     SignalSpec('recorded_async_poll_lane_us', 'recorded_async_poll_lane_us', 'lane_us'),
     SignalSpec('recorded_async_poll_union_us', 'recorded_async_poll_union_us', 'us')), max_facts=14)
QUESTIONS = {s.question_id: s for s in (TASK_COUNT, RUNTIME_SPAN, PMU_CYCLES, COMPILER_HINTS,
    BUFFER_PEAK, PIPELINE_TIME, CLEAN_LATENCY, PHYSICAL_WORK, CRITICAL_WORK, ENGINE_OCCUPANCY,
    DISPATCH_DELAYS, MEMORY_HEADROOM, PMU_ACTIVITY, PMU_COVERAGE, RANK_BALANCE, HOST_COST,
    CLEAN_HOST_COST, CLEAN_PROCESSES, TASK_BOUNDARIES, MEASURED_HOST, ASYNC_POLLING)}


TOPICS = {
    'timing': ('clean', (CLEAN_LATENCY, MEASURED_HOST, CLEAN_PROCESSES, CLEAN_HOST_COST), ('samples', 'measured_host', 'process_rounds', 'host')),
    'work': ('runtime', (PHYSICAL_WORK, CRITICAL_WORK, ENGINE_OCCUPANCY, DISPATCH_DELAYS, RUNTIME_SPAN, TASK_COUNT, RANK_BALANCE, HOST_COST), ('tasks', 'critical_path', 'engine_occupancy', 'dispatch_delays', 'execution_work')),
    'pmu': ('pmu', (PMU_ACTIVITY, PMU_COVERAGE, PMU_CYCLES), ('pmu_samples',)),
    'memory': ('compiler', (MEMORY_HEADROOM, BUFFER_PEAK), ('resources', 'allocations')),
    'compiler': ('compiler', (COMPILER_HINTS, PIPELINE_TIME), ('hints', 'pipeline')),
    'dataflow': ('runtime', (TASK_BOUNDARIES, ASYNC_POLLING), ('boundaries', 'scheduler_phases')),
}


def topic_index():
    """Small default directory; full schemas and optional views are on demand."""
    return [{'topic': topic, 'modes': [mode], 'question': specs[0].question_id}
            for topic, (mode, specs, views) in TOPICS.items()]


def catalog(topic):
    mode, specs, views = TOPICS[topic]
    by_id = {q['question']: q for q in public_questions()}
    questions = [{k: v for k, v in by_id[s.question_id].items() if k != 'detail_views'} for s in specs]
    return {'topic': topic, 'mode': mode, 'questions': questions, 'detail_views': list(views),
            'interpretation': 'observations_and_deltas_only',
            'availability': 'depends_on_saved_native_fields',
            'optional_questions': [s.question_id for s in specs if s in
                (CLEAN_HOST_COST, CLEAN_PROCESSES, HOST_COST, PIPELINE_TIME, ASYNC_POLLING)],
            'notes': {
                'timing': 'Native clean timings and formal scores retain their original meaning. measured_host uses saved BenchmarkStats, excludes warmup and preserves native aggregation. Process references are capture-local, not certified device ranks. Optional host invocation views include warmup.',
                'work': 'Task references are capture-local. Row delay sums are core_us, not elapsed or causal wait time. Use members for each input or rank dispatch; execution_work counts complete members without adding times.',
                'pmu': 'Coverage accompanies PMU summaries. Incomplete coverage prevents total-work comparisons. Pipeline fractions can overlap; activity does not establish a bandwidth limit.',
                'memory': 'Allocation locations refer to generated IR. Capacity comes only from native reports. Unknown allocations or capacity prevent complete headroom conclusions.',
                'compiler': 'Hints reproduce native messages in ordered fragments, with paths redacted. Source references identify native-reported files; they do not certify a mapping to Candidate lines. Pipeline elapsed may include execution.',
                'dataflow': 'Recorded tensor views do not prove copies; logical bytes are not traffic. Polling counts and lane interval unions do not establish DMA waits.',
            }[topic]}


def public_questions():
    """Expose registration vocabulary, keeping query plans internal."""
    details = {'runtime': ['execution_work', 'tasks', 'critical_path', 'engine_occupancy', 'dispatch_delays', 'scheduler_phases', 'boundaries', 'host'], 'pmu': ['pmu_samples', 'pmu'],
               'compiler': ['resources', 'allocations', 'hints', 'pipeline'], 'clean': ['samples', 'host', 'process_rounds','measured_host']}
    return [{'question': spec.question_id, 'modes': list(spec.required_modes),
             'description': spec.owner_question,
             'detail_views': ['device_identity', *details[spec.required_modes[0]]],
             'interpretation': 'performance' if spec == CLEAN_LATENCY else 'diagnostic_observation',
             'expected_signals': list(spec.expected_signal_keys)}
            for spec in QUESTIONS.values()]
