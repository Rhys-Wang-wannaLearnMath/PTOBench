# S0：基线检查与公共测试修缮

阶段：S0。日期：2026-10-02。

结论：**S0 公共基线修缮 PASS，满足进入 S1 的条件。Case 资产存在 BLOCKED 项，未将它们记为通过。**

依据：已按顺序阅读 `Benchmark升级/plan/README.md`、`08_execution_steps.md`，并阅读该目录全部设计文档。本次执行 `00_baseline_and_scope.md` 的 S0；S1～S7 未实施，未运行 NPU 评测或 profiling。S7 仍是升级后的必需全量验收阶段。

## 当前版本与工作区

- Git revision：`9c5dfec2dfa18f056098000123e10de06de32426`。本工作区有大量已有暂存、未暂存和未跟踪内容，实际测试对象是当前工作区，不能仅以 HEAD 代表当前实现。
- 操作前完整状态：[git_status_before.txt](S0_evidence/git_status_before.txt)；原 index：[git_index_before.txt](S0_evidence/git_index_before.txt)；操作后状态：[git_status_after.txt](S0_evidence/git_status_after.txt)。未执行 reset、clean、checkout、暂存或提交。
- 正式 Python：`.venv-official/bin/python`，实际为 CPython 3.10.20、aarch64；安装包版本包括 torch `2.6.0+cpu`、PyPTO `0.1.0`、numpy `2.2.6`。未使用普通 Python 代替正式环境。
- 工具路径及全部 Case environment JSON 保存于 [environment.json](S0_evidence/environment.json)。版本查询见 [toolchain_versions.json](S0_evidence/toolchain_versions.json)：Codex `0.159.2`、PTOAS `0.60`、CANN bisheng/clang `15.0.5`；PTOAS 路径 `/data1/home/ziruiwang/ptoas-v0.60/bin/ptoas`，CANN setup 路径 `/usr/local/Ascend/cann-9.0.0/cann/set_env.sh`。
- PyYAML 尚未安装；按规划属于 S5 的依赖准备，不是本次 CPU 回归的依赖。未将工具路径存在或 CPU 测试结果解释为 NPU 可用性验收。

规划中的 189 个 Case / 153 ready / 12 项测试是此前记录。本次实测为 **192 个 Case、156 ready、36 pending、14 项测试**，未固定旧数量作为断言。

## 实际修改

实现修改仅发生在公共测试 [test_case_comparators.py](../../evaluator/tests/test_case_comparators.py)，原文件及精确差异分别保存为 [test_case_comparators.before.py](S0_evidence/test_case_comparators.before.py) 和 [test_case_comparators.patch](S0_evidence/test_case_comparators.patch)。此外新增本报告、原始证据、资产清单及只读清单采集脚本；没有新增 Bench runtime 功能。

1. 测试内明确适配 `_compare`、`_make_compare`、`make_compare` 三种实际构造入口，支持 `(case_dir, case)` 与 `(case_dir, case, config)`。入口缺失、歧义及未知签名直接失败，不 catch 后跳过。需要时执行 canonical public config；`runpy.run_path` 正确处理带 postponed annotations 的 dataclass，并恢复临时模块登记。
2. 使用按 Case、output、example 划分的 subTest，使某个错误不阻止其余 Case 的检查；任何 subTest 失败仍使整套测试非零退出。
3. 保留标准方法的正反例、Case-local ownership、输出覆盖、callback 不接受 `rtol/atol` 参数等检查。增加 ready certification 与当前 comparator SHA-256 一致性检查，并用宽松调用端容差验证 public policy 不被覆盖。
4. 删除自定义比较器的无条件 `continue`。为全部 8 类自定义规则提供 callback ABI 所需的输出 dtype/rank、paired outputs、INT32 expert counts、slot mapping、chunk lengths/offsets。按 config/policy 保留 top-k 长度、offset、receive capacity 和 EP2/EP4 metadata 行数；与比较无关的 feature/batch 维度缩小，测试不加载巨量正式 Fixture、不执行 kernel。
5. 自定义正反例还覆盖 tie permutation、有效/无效 receive rows、active/inactive mapped rows、缺失绑定 metadata、未选择 chunk rows、INT8 LSB 边界。未知自定义规则明确失败。
6. 全部现存 evaluator 均检查，包括 pending 中的现存实现；ready 缺 evaluator 必须失败。9 个缺 evaluator 的非 ready Case 明确输出 `NOT_RUN` 并进入 pending 资产清单，不将其计入 comparator 执行成功数。

实际构造 **183 个 Case 的 486 个 callback**：标准方法 435 个，自定义方法 51 个。156 个 ready Case 全部覆盖，另覆盖 27 个 pending Case 的 comparator；这不构成 pending Case 的认证或 NPU 正确性结论。

| 自定义规则 | 执行的 callback 数 |
| --- | ---: |
| topk_pair_compare | 9 |
| valid_rows_allclose | 12 |
| ratio_reldiff | 9 |
| int8_lsb | 1 |
| active_recv_ratio_reldiff | 3 |
| valid_ratio_reldiff | 8 |
| indexed_topk_pair | 3 |
| mapped_active_rows_ratio_allclose | 6 |

## 测试命令与实际结果

所有回归均在仓库根目录使用以下命令；保存日志时启用 `pipefail`，没有通过 tee 隐藏失败退出码：

```bash
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python -m unittest discover \
  -s benchmarks/pypto_lib_bench-v0.2/evaluator/tests -p 'test_*.py' -v \
  2>&1 | tee benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S0_evidence/unittest_final_verified.log
```

各次执行仅替换日志文件名，结果均保留：

| 日志 | 退出码 | 实际结果 |
| --- | ---: | --- |
| [unittest_before.log](S0_evidence/unittest_before.log) | 1 | 14 tests，13 通过、1 ERROR；`_compare() missing 1 required positional argument: 'config'` |
| [unittest_after_first.log](S0_evidence/unittest_after_first.log) | 1 | 14 tests，4 个失败 subTest；测试将 EP4 chunk metadata 构造为 `(2, 2)` |
| [unittest_after.log](S0_evidence/unittest_after.log) | 0 | 修正 metadata 行数后，14 tests 全部通过 |
| [unittest_final.log](S0_evidence/unittest_final.log) | 1 | 加强真实 dtype/rank/top-k config 覆盖时，测试的 config loader 未登记 dataclass 的临时模块 |
| [unittest_final_verified.log](S0_evidence/unittest_final_verified.log) | **0** | 最终实现：**Ran 14 tests in 2.068s，OK**；183 个构造入口覆盖，未使用 unittest skip |

前两类追加问题均属于测试输入/加载构造，已在公共测试中修复。最终套件未发现未解决的 comparator 断言失败；未改动任何 canonical comparator 来获得通过。

只读资产清单命令：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python \
  benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S0_evidence/collect_inventory.py
```

结果见 [inventory.log](S0_evidence/inventory.log) 与完整 [inventory.json](S0_evidence/inventory.json)。检查包括认证状态、comparator hash、Fixture manifest/输入目录、全部声明的 Ref output、正数 Ref median、seed/Ref/Oracle 入口及环境/设备要求；输入目录存在性不是完整 Fixture ABI 认证。

## 资产状态、真实缺陷与归属

完整清单：[ready_cases.md](S0_evidence/ready_cases.md)、[pending_cases.md](S0_evidence/pending_cases.md)。

156 个 ready Case 的 comparator 认证 hash 均匹配；其中 **6 个缺少 Candidate Seed 入口**，不能据 ready 状态声称可启动评测。其余 150 个仅通过本次文件存在性/记录检查，未重新认证。

| Case | 当前缺陷 | 状态与归属 |
| --- | --- | --- |
| deepseek_v4.prefill_compressor_ratio4_demo | 缺 `public/candidate_seed/candidate.py` | BLOCKED，Case Candidate Seed 发布 |
| deepseek_v4.prefill_compressor_ratio4_flash | 同上 | BLOCKED，Case Candidate Seed 发布 |
| deepseek_v4.prefill_compressor_ratio4_pro | 同上 | BLOCKED，Case Candidate Seed 发布 |
| deepseek_v4.prefill_layer_ep2_hca_flash | 同上 | BLOCKED，Case Candidate Seed 发布；影响指定 S6 多设备验收 |
| deepseek_v4.qkv_proj_rope_compat_demo | 同上 | BLOCKED，Case Candidate Seed 发布 |
| deepseek_v4.qkv_proj_rope_compat_flash | 同上 | BLOCKED，Case Candidate Seed 发布 |

对上述 6 个 Case 实际运行了维护者 init，输出目录均为 TemporaryDirectory 内的新路径：

```bash
.venv-official/bin/python benchmarks/pypto_lib_bench-v0.2/evaluator/bench.py \
  init CASE_ID --out /tmp/s0_missing_seed_UNIQUE/workspace
```

6 次均退出 **1**，在 `_copy_candidate` 遍历缺失 seed 目录时产生 **FileNotFoundError**。完整 argv、原始错误和逐 Case 日志见 [init_missing_seeds.json](S0_evidence/init_missing_seeds.json)；未执行硬件，也未自动发布或替代缺失 seed。这属于 Case 迁移发布阶段，**不是本次 Bench 升级 S0 的修复范围**；后续 S2 的 ready 资产 preflight 必须拒绝这些 Case。

36 个 pending Case 均保留为未认证资产。清单逐项记录缺失：31 个缺 seed、30 个无有效 Ref latency、15 个缺 Fixture manifest/输入目录、9 个缺 evaluator/Ref/Oracle。缺 evaluator 的 9 个为 `deepseek_v4.moe_ep2_{demo,flash,pro}`、`deepseek_v4.prefill_attention_hca_{demo,flash,pro}`、`deepseek_v4_flash_mtp.prefill_cp_zigzag_cp2_{demo,flash,pro}`。这些缺项归各自 Case 迁移的实现、认证、发布阶段，未据公共 CPU 测试将其改写为 PASS。

## 指定后续真实验收 Case

保留 S6 指定的全部 4 个 Case，未替换 BLOCKED 项。完整环境、源码 hash、comparator hash、tensor 文件路径/大小及 Ref timing 见 [selected_acceptance_cases.json](S0_evidence/selected_acceptance_cases.json)。

| Case | 认证 | seed / Ref 源文件数 | Fixture 输入 / Ref 输出数 | Ref median μs | 设备 | 资产状态 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| deepseek_v4.rmsnorm_decode_flash | ready | 1 / 1 | 2 / 1 | 17.4895 | 0 | 存在性检查通过 |
| deepseek_v4.compressor_ratio4_flash | ready | 1 / 1 | 15 / 4 | 543.99 | 0 | 存在性检查通过；mutating state 保留 |
| deepseek_v4.decode_attention_hca_flash | ready | 6 / 6 | 29 / 1 | 1115.789 | 0 | 存在性检查通过；多文件 helper 闭包保留 |
| deepseek_v4.prefill_layer_ep2_hca_flash | ready | 0 / 20 | 84 / 11 | 16674.7995 | [0, 1] | **BLOCKED：缺 seed** |

前三个 Case 的 seed tree hash 与 Ref tree 在 `ref_npu.py → candidate.py` 映射后逐文件一致。表中耗时来自已有 `ref_performance.json`，不是本次新测量。硬件、真实 profiling 和 Optimizer 验收均尚待对应后续阶段。

## 原有修改及 canonical 保护

操作前后核对结果保存于 [preservation.json](S0_evidence/preservation.json)：

- 操作前已存在的 297 个 tracked 文件内容 SHA-256 全部未改变，原有 index 保留。
- canonical `cases/`、`sealed_assets/`、`environments/` 共 6701 个非缓存文件，未发现新增、删除或变化。
- 其中 3709 个非 `.pt` 文件核对内容 SHA-256，涵盖 Case/config/comparator、Seed、Ref、Oracle、sealed evaluator、认证 JSON 和环境配置，全部一致。
- 2992 个 `.pt` 共约 216.98 GiB，只核对 size、mtime_ns、mode，全部一致；**未对这些 tensor 执行全量字节 hash**。本次所有写操作限于公共测试、报告证据及临时 workspace，没有 Fixture/Ref output 写入或硬件进程。

未修改 Case、Fixture、Ref、Oracle、comparator 或测量语义；未增加 Docker、Supervisor、socket、独立仓库或新发布包。

## S1 前置条件结论

S0 要求的可复现基线、私有构造接口适配、ready comparator 覆盖、完整资产清单、指定验收 Case 摘要及失败证据均已交付；最终公共回归通过。**可以进入 S1 的临时目录文件保护/context/源码复制实现与测试。**

6 个 ready Case 的缺 seed 及 36 个 pending Case 的缺项继续保持 BLOCKED；它们不阻止 S1 的纯 CPU 公共工具工作，但不得进入就绪 Session 或在后续硬件验收中记为 PASS。指定 EP2 Case 的 seed 必须由其 Case 发布负责人补齐后才能完成对应 S6 验收。本次未执行 S1～S7，最终 S7 全量矩阵仍未产生。
