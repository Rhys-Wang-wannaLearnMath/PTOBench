"""Reviewed semantic links from published mechanisms to registered questions."""
from __future__ import annotations


# These mappings indicate which registered observations can inform a card.
# They do not claim that the observation proves the mechanism or its cause.
_REVIEWED = {
    'benchmark_effective_us': {
        'did_clean_latency_change?': ('clean_latency_down',),
        'what_are_measured_host_costs?': ('median_host_us', 'maximum_host_us'),
    },
    'chip_swimlane_dependency_artifacts': {
        'where_is_critical_work?': ('cpm_us', 'observed_path_compute_us', 'observed_path_gap_us'),
        'where_are_dispatch_delays?': ('dispatch_to_receive_core_us', 'receive_to_start_core_us', 'end_to_finish_core_us'),
    },
    'incore_pmu_alignment_and_escape': {
        'which_pipelines_are_active?': ('scalar_cycle_fraction', 'vec_cycle_fraction', 'cube_cycle_fraction',
                                        'mte1_cycle_fraction', 'mte2_cycle_fraction', 'mte3_cycle_fraction'),
        'did_pmu_total_cycles_change?': ('pmu_cycles_down',),
    },
    'memory_address_allocation_and_cross_core_ring': {
        'how_much_local_memory_headroom?': ('minimum_headroom_bytes', 'maximum_space_utilization', 'largest_buffer_bytes'),
    },
    'memory_reuse_and_declared_allocations': {
        'how_much_local_memory_headroom?': ('minimum_headroom_bytes', 'maximum_space_utilization', 'largest_buffer_bytes'),
        'did_buffer_peak_change?': ('buffer_peak_down',),
    },
    'pass_artifacts_and_compiler_diagnostics': {
        'did_compiler_hints_change?': ('compiler_hints_changed',),
        'how_much_local_memory_headroom?': ('minimum_headroom_bytes', 'maximum_space_utilization', 'largest_buffer_bytes'),
    },
    'scope_stats_and_ring_deadlocks': {
        'how_much_async_completion_polling_is_recorded?': ('recorded_async_poll_records', 'recorded_async_poll_lanes',
                                                           'recorded_async_poll_lane_us', 'recorded_async_poll_union_us'),
    },
    'software_pipeline_and_slots': {
        'did_pipeline_elapsed_change?': ('pipeline_elapsed_down',),
        'how_much_local_memory_headroom?': ('minimum_headroom_bytes', 'maximum_space_utilization', 'largest_buffer_bytes'),
    },
    'speculative_early_dispatch': {
        'how_did_dispatched_work_change?': ('logical_tasks', 'physical_rows'),
        'where_is_critical_work?': ('cpm_us', 'observed_path_compute_us', 'observed_path_gap_us'),
    },
    'task_granularity_and_dispatch_overhead': {
        'did_task_granularity_change?': ('task_count_down',),
        'how_did_dispatched_work_change?': ('logical_tasks', 'physical_rows'),
        'how_busy_are_compute_engines?': ('aic_occupancy', 'aiv_occupancy', 'aic_tail_occupancy', 'aiv_tail_occupancy'),
    },
}


def reviewed_bindings(card_ids):
    selected = set(card_ids)
    return {
        'schema_version': 1,
        'bindings': [
            {'mechanism_id': card_id, 'card_ref': 'card:' + card_id,
             'evidence': [{'question_id': question, 'signal_keys': list(signals)}
                          for question, signals in questions.items()]}
            for card_id, questions in sorted(_REVIEWED.items()) if card_id in selected
        ],
    }
