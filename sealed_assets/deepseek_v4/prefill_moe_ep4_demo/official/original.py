# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE end-to-end prefill harness.

Compatibility wrapper for the refreshed distributed/token-major MoE.  The MoE
module freezes its token shape at import time, so prefill must set MOE_TOKENS to
PREFILL_TOKENS before importing it.
"""

import pypto.language as pl  # kept for older importers expecting a PyPTO module

import config
config.MOE_TOKENS = config.PREFILL_TOKENS
config.RECV_MAX = config.EP_WORLD_SIZE * config.MOE_TOKENS

from pypto.ir.distributed_compiled_program import DistributedConfig

from config import PREFILL_BATCH, PREFILL_SEQ
from moe import (
    D,
    HC_DIM,
    HC_MULT,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    TOPK,
    VOCAB,
    T as MOE_T,
    build_tensor_specs as build_moe_tensor_specs,
    golden_moe as golden_prefill_moe,
    l3_moe,
)


B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S

assert T == MOE_T, (
    "prefill_moe must import moe with MOE_TOKENS=PREFILL_TOKENS; "
    f"got prefill T={T}, moe T={MOE_T}"
)

# Backwards-compatible public names.  The current MoE implementation is an L3
# distributed program; keep prefill_moe as an alias so external harnesses can
# still import it from this module.
prefill_moe = l3_moe

# Old symbol aliases for lightweight external importers.
N_EXPERTS = N_EXPERTS_GLOBAL
N_LOCAL_EXPERTS = N_LOCAL


def build_tensor_specs(layer_id=0, num_tokens=T):
    return build_moe_tensor_specs(layer_id=layer_id, num_tokens=num_tokens)


if __name__ == "__main__":
    import argparse

    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
                        help="EP world size / rank count; parsed by moe at import time")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids (need {N_RANKS})")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=T,
                        help=f"active token count for MoE dispatch/combine (0..{T})")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) == N_RANKS, f"need exactly {N_RANKS} devices, got {device_ids}"

    result = run_jit(
        fn=l3_moe,
        specs=build_tensor_specs(layer_id=args.layer_id, num_tokens=args.num_tokens),
        golden_fn=golden_prefill_moe,
        golden_data=args.golden_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids,
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
            log_level=args.log_level,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "x_next": ratio_reldiff(diff_thd=3e-3, pct_thd=0.05),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
