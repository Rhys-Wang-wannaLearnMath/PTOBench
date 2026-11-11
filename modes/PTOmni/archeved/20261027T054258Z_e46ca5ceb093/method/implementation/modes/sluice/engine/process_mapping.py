"""Bind measured chip processes using existing, complete SDK startup records.

The device is the runtime-visible ID passed to ChipWorker.init. Rank is its
position in the archived local device declaration, not a global HCCL rank.
No process ordering, clock subtraction or extra collection is used.
"""
import re

BASIS = 'sdk_ready_declared_local_device_order_v1'
_READY = re.compile(r'\[chip_process pid=([1-9][0-9]*) dev=(0|[1-9][0-9]*)\] ready')


def bind(lines, device_ids, measured_pids):
    """Optional mapping: malformed identity must not discard valid timings."""
    from profile_db._offline_tools.strace_timing import parse_spans
    if device_ids is None:
        return {}, 'clean_process_mapping_device_declaration_missing'
    if (not isinstance(device_ids,list) or not device_ids
            or any(type(i) is not int or i < 0 for i in device_ids)
            or len(set(device_ids))!=len(device_ids)):
        return {}, 'clean_process_mapping_device_declaration_invalid'
    ready, first_run = {}, {}
    for index,line in enumerate(lines):
        if '[chip_process' in line:
            match = _READY.fullmatch(line.strip())
            if match is None:
                return {}, 'clean_process_mapping_record_invalid'
            pid,device = map(int,match.groups())
            if pid in ready:
                return {}, 'clean_process_mapping_process_repeated'
            ready[pid] = (device,index)
        for span in parse_spans([line]):
            if span.name=='chip.run' and span.depth==0 and span.pid in measured_pids:
                first_run.setdefault(span.pid,index)
    if not ready:
        return {}, 'clean_process_mapping_startup_missing'
    if set(ready)!=set(measured_pids) or set(first_run)!=set(measured_pids):
        return {}, 'clean_process_mapping_process_coverage_mismatch'
    if (len(ready)!=len(device_ids)
            or {device for device,_ in ready.values()}!=set(device_ids)):
        return {}, 'clean_process_mapping_device_coverage_mismatch'
    if any(position>=first_run[pid] for pid,(_,position) in ready.items()):
        return {}, 'clean_process_mapping_lifetime_ambiguous'
    return {pid:{'device':device,'rank':device_ids.index(device)}
            for pid,(device,_) in ready.items()}, None
