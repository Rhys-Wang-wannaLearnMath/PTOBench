# PTOBench 20-Case READY stratified selection (2026-10-11)

This subset selects the highest-ranked READY Cases from the existing 40-Case list, preserving equal `Standalone` / `Composite` strata (10 each). All selected Cases are currently `READY`; the selection favors visible memory reuse, dependency scheduling, buffering, tiling and distributed orchestration headroom.

| # | Case ID | Class | Status | Optimization basis |
| ---: | --- | --- | --- | --- |
| 1 | `deepseek_v4.compressor_ratio4_demo` | **Standalone** | `READY` | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 2 | `deepseek_v4.compressor_ratio4_flash` | **Standalone** | `READY` | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 3 | `deepseek_v4.compressor_ratio4_pro` | **Standalone** | `READY` | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 4 | `deepseek_v4.prefill_compressor_ratio4_demo` | **Standalone** | `READY` | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 5 | `deepseek_v4.prefill_compressor_ratio4_flash` | **Standalone** | `READY` | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 6 | `deepseek_v4.prefill_compressor_ratio4_pro` | **Standalone** | `READY` | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 7 | `deepseek_v4.prefill_qkv_proj_rope_v2_decode_demo` | **Standalone** | `READY` | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 8 | `deepseek_v4.prefill_qkv_proj_rope_v2_decode_flash` | **Standalone** | `READY` | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 9 | `deepseek_v4.prefill_qkv_proj_rope_v2_decode_pro` | **Standalone** | `READY` | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 10 | `deepseek_v4_flash_mtp.prefill_cp_zigzag_cp2_demo` | **Standalone** | `READY` | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 11 | `deepseek_v4_flash_dspark.prefill_layer_csa_demo` | **Composite** | `READY` | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 12 | `deepseek_v4_flash_dspark.prefill_layer_csa_flash` | **Composite** | `READY` | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 13 | `deepseek_v4_flash_dspark.prefill_layer_hca_pro` | **Composite** | `READY` | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 14 | `deepseek_v4_flash_dspark.prefill_csa_flash` | **Composite** | `READY` | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 15 | `deepseek_v4_flash_dspark.prefill_csa_pro` | **Composite** | `READY` | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 16 | `deepseek_v4_flash_dspark.prefill_hca_flash` | **Composite** | `READY` | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 17 | `deepseek_v4_flash_dspark.prefill_hca_pro` | **Composite** | `READY` | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 18 | `deepseek_v4_flash_dspark.decode_swa_flash` | **Composite** | `READY` | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 19 | `deepseek_v4_flash_dspark.decode_swa_pro` | **Composite** | `READY` | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 20 | `deepseek_v4.prefill_attention_csa_flash` | **Composite** | `READY` | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |

## Stratum counts

| Class | READY | Total |
| --- | ---: | ---: |
| Standalone | 10 | 10 |
| Composite | 10 | 10 |
| **Total** | **20** | **20** |

Source: [`CASE_SELECTION_40_20261011.json`](CASE_SELECTION_40_20261011.json).
