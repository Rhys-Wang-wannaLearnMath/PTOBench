# PTOBench 40-Case stratified selection (2026-10-11)

The selection is stratified 20 `Standalone` / 20 `Composite`. It favors attention/layer/MoE orchestration, compression, QKV projection and distributed CP exchange, where memory reuse, dependency scheduling and tiling have visible optimization headroom. Status distinguishes already READY Cases from Cases still in the migration queue.

| # | Case ID | Class | Status | Stage | Why selected |
| ---: | --- | --- | --- | --- | --- |
| 1 | `deepseek_v4.compressor_ratio4_demo` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 2 | `deepseek_v4.compressor_ratio4_flash` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 3 | `deepseek_v4.compressor_ratio4_pro` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 4 | `deepseek_v4.prefill_compressor_ratio4_demo` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 5 | `deepseek_v4.prefill_compressor_ratio4_flash` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 6 | `deepseek_v4.prefill_compressor_ratio4_pro` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 7 | `deepseek_v4.prefill_qkv_proj_rope_v2_decode_demo` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 8 | `deepseek_v4.prefill_qkv_proj_rope_v2_decode_flash` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 9 | `deepseek_v4.prefill_qkv_proj_rope_v2_decode_pro` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 10 | `deepseek_v4_flash_mtp.prefill_cp_zigzag_cp2_demo` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 11 | `deepseek_v4_flash_mtp.prefill_cp_zigzag_cp2_flash` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 12 | `deepseek_v4_flash_mtp.prefill_cp_zigzag_cp2_pro` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 13 | `deepseek_v4.prefill_qkv_proj_rope_v2_prefill_demo` | **Standalone** | `migrating` | Stage 6 accept (QUEUED) | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 14 | `deepseek_v4.prefill_qkv_proj_rope_v2_prefill_flash` | **Standalone** | `migrating` | Stage 6 accept (QUEUED) | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 15 | `deepseek_v4.prefill_qkv_proj_rope_v2_prefill_pro` | **Standalone** | `migrating` | Stage 6 accept (QUEUED) | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 16 | `deepseek_v4_flash_mtp.prefill_cp_zigzag_cp4_demo` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 17 | `deepseek_v4_flash_mtp.prefill_cp_zigzag_cp4_flash` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 18 | `deepseek_v4_flash_mtp.prefill_cp_zigzag_cp4_pro` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 19 | `deepseek_v4_flash_mtp.prefill_cp_zigzag_cp8_demo` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 20 | `deepseek_v4_flash_mtp.prefill_cp_zigzag_cp8_flash` | **Standalone** | `ready` | READY | memory traffic/cache layout and projection or collective exchange are exposed, giving direct memory-planner and tiling/search space. |
| 21 | `deepseek_v4_flash_dspark.prefill_layer_csa_demo` | **Composite** | `ready` | READY | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 22 | `deepseek_v4_flash_dspark.prefill_layer_csa_flash` | **Composite** | `ready` | READY | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 23 | `deepseek_v4_flash_dspark.prefill_layer_hca_pro` | **Composite** | `ready` | READY | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 24 | `deepseek_v4_flash_dspark.prefill_csa_flash` | **Composite** | `ready` | READY | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 25 | `deepseek_v4_flash_dspark.prefill_csa_pro` | **Composite** | `ready` | READY | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 26 | `deepseek_v4_flash_dspark.prefill_hca_flash` | **Composite** | `ready` | READY | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 27 | `deepseek_v4_flash_dspark.prefill_hca_pro` | **Composite** | `ready` | READY | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 28 | `deepseek_v4_flash_dspark.decode_swa_flash` | **Composite** | `ready` | READY | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 29 | `deepseek_v4_flash_dspark.decode_swa_pro` | **Composite** | `ready` | READY | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 30 | `deepseek_v4.prefill_attention_csa_flash` | **Composite** | `ready` | READY | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 31 | `deepseek_v4_flash_dspark.decode_layer_csa_demo` | **Composite** | `migrating` | Stage 3 implement-certify (RUNNING source) | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 32 | `deepseek_v4_flash_dspark.decode_layer_csa_flash` | **Composite** | `migrating` | Stage 3 implement-certify (RUNNING source) | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 33 | `deepseek_v4_flash_dspark.decode_layer_hca_demo` | **Composite** | `migrating` | Stage 3 implement-certify (RUNNING source) | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 34 | `deepseek_v4_flash_dspark.decode_layer_hca_flash` | **Composite** | `migrating` | Stage 3 implement-certify (RUNNING source) | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 35 | `deepseek_v4_flash_dspark.decode_layer_hca_pro` | **Composite** | `migrating` | Stage 3 implement-certify (RUNNING source) | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 36 | `deepseek_v4_flash_dspark.decode_layer_swa_demo` | **Composite** | `migrating` | Stage 3 implement-certify (RUNNING source) | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 37 | `deepseek_v4_flash_dspark.decode_layer_swa_flash` | **Composite** | `migrating` | Stage 3 implement-certify (RUNNING source) | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 38 | `deepseek_v4.prefill_layer_ep8_hca_demo` | **Composite** | `migrating` | Stage 2 contract (QUEUED) | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 39 | `deepseek_v4.prefill_layer_ep8_hca_flash` | **Composite** | `migrating` | Stage 2 contract (QUEUED) | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |
| 40 | `deepseek_v4.prefill_moe_ep8_demo` | **Composite** | `migrating` | Stage 3 implement-certify (QUEUED) | multi-stage attention/MoE/layer orchestration exposes fusion, dependency, buffering and distributed scheduling search space. |

## Stratum counts

| Class | READY | Migrating/maintenance/legacy | Total |
| --- | ---: | ---: | ---: |
| Standalone | 17 | 3 | 20 |
| Composite | 10 | 10 | 20 |
