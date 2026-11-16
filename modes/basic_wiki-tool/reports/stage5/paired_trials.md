# PTOAnalyzer Stage 5 — paired Session report

All paths below are under `/data2/ziruiwang/PTOBench/stage5_trials/runs/`. The six primary Session roots completed normally through PTOBench; raw event, authority, Agent and finish archives remain in each root. Candidate Diff counts compare the selected final candidate tree to the canonical Candidate copied into Event 1.

## Profile coverage

The authoritative audit covers **60 Formal + 15 Held-out = 75/75** Cases. Every selected Profile uses the canonical starting Candidate and records a completed combined Event, its Session/Event ID, candidate hash, device identities, authorized artifact index and diagnostic coverage. Each Case has exactly one selected record. The 94 references comprise 75 selected records, 18 usable duplicate references not selected by the stated preference, and one unusable original Profile retained as failure evidence. Unselected originals were not deleted. The selection rule prefers canonical Candidate matches and then the latest usable authorized completed Profile.

## Matched sessions

Pair conditions were matched per Case: same effective model `provider-default/deepseek-flash` / `high`, Pi 1.5.2, seed, canonical starting Candidate hash, Case/environment/configuration and device group. Orders 1–4 used the original 20-Evaluate limit; the two remaining CP8 arms used 10. Profile events are reported separately and do not consume that Evaluate allowance. All runs used normal PTOBench Session/device locking.

| Order / arm | Case | Session root | Devices | Evaluate used / limit; Profiles; finish Event | Final correctness / Clean Latency | Selected Candidate SHA-256 | Candidate Diff from canonical start |
|---|---|---|---:|---:|---|---|---|
| 1 / `basic` | `deepseek_v4.lm_head_tp2_flash` | `s5_01_lm_head_basic/deepseek_v4.lm_head_tp2_flash/20261115T013039Z_41d365f68093` | `[0, 1]` | 12/20; 5; Event 19 | passed / 802.3995 µs | `3f626bbdf9ed9ba09d811884ba64526145d31f9e6e76a0a9ac74371f47934baa` | `candidate.py` (+104/−73), `variants_831.py.txt` (+179/−0) (283 added / 73 deleted) |
| 2 / `basic_wiki-tool` | `deepseek_v4.lm_head_tp2_flash` | `s5_02_lm_head_wiki_tool/deepseek_v4.lm_head_tp2_flash/20261115T084928Z_e7f77e99ccad` | `[0, 1]` | 18/20; 5; Event 24 | passed / 877.5800 µs | `b534d53b0acb567c43e63461654f0f0c3426cc21811905d73564073adae1d410` | `candidate.py` (+87/−104) (87 added / 104 deleted) |
| 3 / `basic_wiki-tool` | `deepseek_v4.prefill_layer_ep2_csa_demo` | `s5_03_prefill_layer_wiki_tool/deepseek_v4.prefill_layer_ep2_csa_demo/20261115T093220Z_20f5f4546e4c` | `[0, 1]` | 11/20; 24; Event 36 | passed / 9,192.2895 µs | `7f783e5c9f93ba2f5a10a5a86d2c473c3cd89ddc397b66fce52ee62d5fe0d624` | `candidate.py` (+66/−0), `gate.py` (+12/−5), `moe.py` (+33/−20), `prefill_compressor_ratio4.py` (+14/−18), `prefill_hc_pre.py` (+202/−205), `prefill_indexer_compressor.py` (+11/−13) (338 added / 261 deleted) |
| 4 / `basic` | `deepseek_v4.prefill_layer_ep2_csa_demo` | `s5_04_prefill_layer_basic/deepseek_v4.prefill_layer_ep2_csa_demo/20261115T123730Z_e39c5644a361` | `[0, 1]` | 14/20; 10; Event 25 | passed / 18,545.7100 µs | `bb94e977ba038e56b6f571c270eae5c810d50665de00e0f198d1d77b617ea931` | `_agg.py` (+24/−0), `_tl.py` (+38/−0), `_tl2.py` (+27/−0), `expert_routed.py` (+22/−10), `expert_shared.py` (+19/−10), `gate.py` (+39/−34), `prefill_attention_csa.py` (+5/−4), `prefill_compressor_ratio4.py` (+34/−31), `prefill_hc_pre.py` (+219/−217), `prefill_indexer_compressor.py` (+34/−26) (461 added / 332 deleted) |
| 5 / `basic` | `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` | `s5_05_cp8_basic/deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo/20261115T140928Z_d69e0bda1447` | `[0, 1, 2, 3, 4, 5, 6, 7]` | 9/10; 14; Event 24 | passed / 7,204.3100 µs | `921c437b1bf4b6bbc4d9d70135babbdc7d671a6248ca3c6f63c98656efc89c59` | `_ckpt1_candidate.txt` (+767/−0), `candidate.py` (+245/−170), `prefill_cp_exchange.py` (+16/−9) (1028 added / 179 deleted) |
| 6 / `basic_wiki-tool` | `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` | `s5_06_cp8_wiki_tool/deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo/20261115T160157Z_8a2c270c10e6` | `[0, 1, 2, 3, 4, 5, 6, 7]` | 10/10; 3; Event 14 | passed / 6,518.1095 µs | `e8a3c9dc488d0f60b4a5e8cda543047033bb2092f2adda929fc7990e657b6fef` | `candidate.py` (+95/−100), `prefill_cp_exchange.py` (+64/−61), `tuning.json` (+2/−2) (161 added / 163 deleted) |

Each session has `events.jsonl`, `final_result.json`, `launcher_result.json` and `final/authority_result.json` at the listed root. `summary.json` includes all Event rows with action, state, Candidate hash, Profile/archive reference, device group where available, integrity and failure status. Compile/correctness failures are preserved as Event evidence; only the final official finish determines the reported final Clean Latency.

## Pair outcomes

| Case | Basic | Wiki + PTOAnalyzer | Treatment delta vs Basic | Observation |
|---|---:|---:|---:|---|
| `deepseek_v4.lm_head_tp2_flash` | 802.3995 µs | 877.5800 µs | +9.37% | treatment slower; both final correctness PASS |
| `deepseek_v4.prefill_layer_ep2_csa_demo` | 18,545.7100 µs | 9,192.2895 µs | -50.43% | treatment lower; both final correctness PASS |
| `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` | 7,204.3100 µs | 6,518.1095 µs | -9.52% | treatment lower; both final correctness PASS |

These are descriptive results for Wiki + PTOAnalyzer as one treatment. The experiment does not establish which component contributed to a latency change or generalize beyond these three pairs.

## Agent tools and Wiki

| Treatment Session | Registration | Analyzer invocations | Outcomes | Wiki identity / read-only check | Later Analyzer-result use |
|---|---:|---|---|---|---|
| `20261115T084928Z_e7f77e99ccad` | 9/9 active | `compute_unit_activity` (success, Profile Event 1), `compute_task_share` (success, Profile Event 1), `task_distribution_check` (success, Profile Event 1) | errors=0; cancelled=0; interrupted=0 | SHA-256 `f8df803c68c05ba326a52eaf5d1129cb19bdc6ba9c0acddfe45e3241f3abcf3f` (INDEX `30256aba51afe2631ef6bddc271bf8c971054be1b27e894429f71f2c8eb9b23e`); unchanged=True; Agent read refs=28 | Not established by explicit durable records |
| `20261115T093220Z_20f5f4546e4c` | 9/9 active | `compute_unit_activity` (success, Profile Event 1) | errors=0; cancelled=0; interrupted=0 | SHA-256 `f8df803c68c05ba326a52eaf5d1129cb19bdc6ba9c0acddfe45e3241f3abcf3f` (INDEX `30256aba51afe2631ef6bddc271bf8c971054be1b27e894429f71f2c8eb9b23e`); unchanged=True; Agent read refs=8 | Not established by explicit durable records |
| `20261115T160157Z_8a2c270c10e6` | 9/9 active | `compute_unit_activity` (success, Profile Event 1) | errors=0; cancelled=0; interrupted=0 | SHA-256 `f8df803c68c05ba326a52eaf5d1129cb19bdc6ba9c0acddfe45e3241f3abcf3f` (INDEX `30256aba51afe2631ef6bddc271bf8c971054be1b27e894429f71f2c8eb9b23e`); unchanged=True; Agent read refs=9 | Not established by explicit durable records |

All nine PTOAnalyzer names were registered and active in each of the three treatment Sessions. There were five successful Analyzer invocations total and no Analyzer-call error, cancellation or interruption. Invocation is not evidence of later use; the available records do not tie a returned result to a subsequent Candidate decision. Pi shell-command failures and Candidate compile/correctness failures are recorded separately in `summary.json` and each Agent/Event archive.

## Cost, termination and recovery records

| Session | Wall clock (s) | Input / cached input / output tokens | Stop reason |
|---|---:|---:|---|
| `20261115T013039Z_41d365f68093` | 2671.420 | 178,058 / 23,462,528 / 180,696 | `optimizer_exit` (CLI exit 0) |
| `20261115T084928Z_e7f77e99ccad` | 2482.804 | 262,956 / 17,337,856 / 78,464 | `optimizer_exit` (CLI exit 0) |
| `20261115T093220Z_20f5f4546e4c` | 8484.893 | 351,772 / 45,134,720 / 150,496 | `optimizer_exit` (CLI exit 0) |
| `20261115T123730Z_e39c5644a361` | 4720.553 | 221,050 / 28,803,200 / 120,736 | `optimizer_exit` (CLI exit 0) |
| `20261115T140928Z_d69e0bda1447` | 5865.322 | 251,384 / 32,485,632 / 139,059 | `optimizer_exit` (CLI exit 0) |
| `20261115T160157Z_8a2c270c10e6` | 3518.669 | 730,461 / 70,193,408 / 89,444 | `optimizer_exit` (CLI exit 0) |

All six primary Sessions ended with `optimizer_exit`, CLI exit 0 and a passing official finish. Across them, 74 accepted Evaluate events, 61 combined Profile events and six finish actions are recorded; one additional Evaluate request in Trial 1 was rejected and did not consume the accepted budget. The two CP8 arms used 9/10 and 10/10 accepted Evaluates.

Additional attempts, excluded from the six primary pair rows, are preserved:
- `s5_02_lm_head_wiki_tool/deepseek_v4.lm_head_tp2_flash/20261115T021536Z_aecc178a0ae4` — failed/optimizer_nonzero_exit, 0 hardware actions; Launcher failed before Pi/Agent startup with zero hardware actions; one corrected standard-CLI launch produced the paired Session.
- `s5_04_prefill_layer_basic/deepseek_v4.prefill_layer_ep2_csa_demo/20261115T115549Z_45c93d6096f8` — incomplete interrupted precursor, 9 hardware actions; `session.json` retains `running` but final process/lock inspection found no active owner or action. Its pending fork reservation remains preserved; there was no official finish. The paired Basic retry completed under the planned Case/config.
- `s5_06_cp8_wiki_tool/deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo/20261115T155046Z_ca4ac0cb3a77` — failed/optimizer_nonzero_exit, 0 hardware actions; Launcher failed before Pi/Agent startup with zero hardware actions; one corrected standard-CLI launch produced the paired Session.

## Verification

- Mode CPU suite: `PYTHONPATH=.:evaluator:evaluator/tests /data1/home/ziruiwang/pypto-lib/.venv-official/bin/python -m pytest modes/basic_wiki-tool/tests -q` — 11 passed.
- Public Pi interface: `PYTHONPATH=.:evaluator:evaluator/tests /data1/home/ziruiwang/pypto-lib/.venv-official/bin/python -m pytest evaluator/tests/test_isolation.py -k 'pi_tool_' -q` — 2 passed, 26 deselected, 9 subtests passed.
- These passed before report finalization and were not rerun because implementation code did not change. JSON report files were parsed and `git diff --check` passed.
- No public PTOBench, Basic or CANNBot implementation was changed. The local Unix Socket path fix is exercised by the completed treatment Sessions through the unchanged standard isolation path.
