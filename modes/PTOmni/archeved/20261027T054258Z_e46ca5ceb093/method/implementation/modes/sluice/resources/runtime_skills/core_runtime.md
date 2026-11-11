# SLUICE 核心运行规则

这是当前 Run 冻结的工作合同。当前 Case、设备、固定接口、源码范围、tuning 权限和可用工具以
workspace 的 `.benchmark_context/` 与工具 schema 为准；本合同不复述这些动态事实。若合同与
workspace 或 Controller 的当前状态冲突，遵循 workspace 与工具返回值。

## 工作协议

1. 每次决策前调用 `sluice_progress`，按当前现象用 `sluice_search` 查找相关知识；只读实际返回
   的引用，并用 `sluice_read_skill` 读完整内容和依赖。搜索不到的内容保持未知，不从旧文档、Git
   或记忆补造 API。
2. 每次只推进一个由证据或源码支持、可验证的方向。若 `sluice_progress` 没有已接受的下一步
   Decision（包括首次决策及每个周期完成后），先用当前 `state_revision` 调用 `sluice_step`，提交
   下一 Decision 和所需机会；解释已完成周期时一并提交其 `completed_cycle_id`。Controller 接受
   `optimize` 或 `recheck` 后，使用返回的 revision 调用 `sluice_register`，登记当前 parent source、
   范围、前置条件、预期信号、验证问题、回退条件和知识引用，然后才按登记范围修改 Candidate。
   `completed_cycle_id` 必须逐字取自本次 `sluice_progress` 返回且状态为 `feedback_ready` 或
   `completed` 的周期；没有符合项就省略，不要推测或复用旧 ID。若 `sluice_step` 或
   `sluice_register` 被拒绝，不要重复同一请求；重新读取 `sluice_progress`，按新的 revision 和
   Controller 当前允许的状态修正一次。`stop` Decision 用 `sluice_finish` 收尾。一次只能有一个活动 Hypothesis。
3. 只修改登记的范围。修改后从 `sluice_progress` 取得当前 source identity，再调用
   `sluice_evaluate`。编译失败、运行失败、结果错误、正确但变慢是不同结果，均须保留；不能把失败
   解释成没有性能变化。
4. 仅当某个已登记问题的答案可能改变下一步时调用 `sluice_profile`。每个问题单独形成
   EvidenceBundle；只能使用该问题允许的信号。没有数据是 unavailable，不是零，也不是 no gain。
5. 重新读取当前 ProgressView 和 Decision，再决定继续、诊断、重新检查或结束。Controller 管理
   假设、尝试、预算、回退和 best；不得另建状态、记忆或预算文件，不得自行写入事实或成绩。

## 状态引用与预算

- `sluice_progress.progress.cycles[].cycle_id` 是状态接口中的 cycle ID。当前请求上下文里的
  `recent_cycle.cycle_id` 和 `cycle_refs` 可能是 `cycle:<opaque>` 引用；`sluice_step` 接受这两种
  对应值。只可引用本 Session 中 status 为 `feedback_ready` 或 `completed` 的周期；若没有，省略
  `completed_cycle_id`。不要把 hypothesis、program 或 event 引用用作 cycle 引用。
- `sluice_search` 优先使用 query；只有拿到知识包当前返回的 topic 名称时才传 `topic`。知识 topic
  与 profile catalog topic 是不同命名空间。Profile 问题必须逐字取自相关 `sluice_profile_catalog`
  返回的注册问题；profile detail 只使用该 catalog 列出的 view。
- `sluice_step` 的 `optimize`、`recheck`、`diagnose` Decision 都必须引用已提交或已存在的 opportunity；
  `diagnose` 还必须提供 catalog 中注册的 `evidence_question`。diagnose 不会创建 Hypothesis 或 cycle。
  对未注册 Hypothesis 的当前源码基线，使用 `sluice_profile_baseline`，再按 catalog 读取需要的
  `sluice_profile_detail`；不要在此时调用仅服务于 evaluate 后 Hypothesis 的 `sluice_profile`。基线
  诊断结束后重新读取 progress，并再次 `sluice_step` 提交 optimize、recheck 或 stop Decision。
- 每次查看 `remaining_hardware_actions`。开始一个新 Hypothesis 前，为它的 evaluate、尚未完成的
  注册 Evidence 问题和结束动作留出预算。任务要求的最低完整周期数和证据条件达到后，不再开启
  探索周期；提交 stop Decision 并调用 `sluice_finish`，不要等硬件预算耗尽。
- 本次 Stage 01 smoke 的最低验收是两个完整 optimize cycle、一次 clean evaluate、一次真实 profile，
  以及下一次 profile 后复读同一首轮 raw 文件。一个 cycle 只有在登记 Hypothesis 后完成修改、
  `sluice_evaluate` 返回有效且 integrity-passed 的 correctness PASS 结果，并据此读取反馈和提交下一
  Decision 后才计数；编译失败、`action_failed` 或没有有效 evaluate 结果的尝试不计数。正确但变慢的
  有效结果仍是已完成 cycle，须按回退条件和 Controller 状态继续。该 smoke 优先直接形成 optimize
  Decision；只有基线诊断会改变首个 Hypothesis 时才先 diagnose。失败后如剩余预算允许，应根据具体
  编译/运行诊断修正并继续，不要把失败尝试当成两轮完成，也不要因尚有可用预算就提前 finish。只有
  两个有效 cycle 和证据条件都满足，且没有待完成的注册 Evidence，才提交 stop/finish；若预算或安全
  条件确实不允许继续，明确报告未达门槛，不得声称通过。
- 当任务要求检查 raw capture 保持性，首个产生 raw capture 的 profile action 完成后，立即列出并读取
  一个原始文件至完整内容；记录这次成功读取的精确 `profile_ref`、`file_ref` 和内容摘要。在后续一次
  profile action 完成后，使用同一对引用再次读取完整内容并比较摘要。不要先执行后续 profile 再做首次
  读取，也不要换用另一个文件代替复读；此检查不需要额外硬件 action。

## 工具行为与边界

- `sluice_progress` 只读当前状态、程序身份、预算和 Decision。
- `sluice_search` 与 `sluice_read_skill` 只访问已发布知识包。按返回的 cursor/分页规则继续读取；
  内容未读全时不要声称完整理解。
- `sluice_register` 登记修改前的 Hypothesis，不执行修改，也不授予额外权限。
- `sluice_evaluate`、`sluice_profile` 和 `sluice_fork` 通过 PTOBench 公共 action 执行；不要用
  shell、私有接口或自行复制 workspace 代替。
- `sluice_profile` 给出诊断证据，不能替代 correctness 或 clean latency 验证。原始 profile 按需
  只读能力由当前运行配置控制；不要求一律读取或回避读取。
- 若为验证 raw capture 保持性而读取文件，保存第一次成功读取的 `profile_ref` 和 `file_ref`；在后续
  profile action 之后用同一对引用再次只读读取并比较返回内容，不要另选文件或增加硬件 action。
- `sluice_finish` 提交结束原因；正式结果仍由 PTOBench 完成。

遵守 workspace 固定 ABI、输入输出契约、tuning 白名单和设备分配。不得读取 sealed/ref/fixture/
evaluator/control 内容，不得修改 `.benchmark_context`、受保护记录或预算。不得直接执行硬件命令，
不得绕过公共 action，不得创建第二套 memory。工具 schema 不在本文件重复。

## 当前 PyPTO 基础语义

使用当前安装环境中核实过的 `pypto.language` API；单凭文档、名称或可导入性不能证明当前设备支持。
找不到对应当前 API 时，将其标为未知并先检查当前安装。

- `Tensor` 与 `Tile` 是不同层次。`slice` 描述 Tensor 区域，`load` 将 Tensor 数据搬到 Tile，
  `store` 将 Tile 写回 Tensor；不能把 Tensor 专用参数套用到 Tile。
- `valid_shape` 描述每轴有效范围，不能生成缺失数据。尾块、部分写入和 reduction identity 都须
  保持正确。`reshape` 是零拷贝 view，必须保持元素数及可表示的有效区域，不保证 contiguous。
- `pl.range` 表示 DSL 顺序循环，`pl.parallel` 只适用于独立迭代，`pl.unroll` 是常量边界展开；
  不要用 Python 循环机械替换 DSL 循环。展开会影响代码量、寄存器和任务数。
- 当前设备流水线使用 `pl.pipeline(..., stage=正整数)`。它会改变缓冲区生命周期和资源压力，不能
  当作旧的自动 fusion 或 submit 开关。依赖和内存可行性须保持正确。
- 修改 tile 尺寸或 dtype 前，先检索并读取匹配的 alignment/内存知识；依据实际 dtype、布局和编译器
  约束核算连续维度的字节宽度。编译器报告 alignment 错误时，该 Candidate 不算有效 cycle；保留诊断，
  选择满足明确约束的新尺寸后再 evaluate，不要重复失败尺寸或只凭直觉改变另一维。
- Orchestration 与 InCore 职责不同；scope 是依赖和资源生命周期边界。不要为减少表面等待而删除
  依赖、使用未经授权的 `no_dep` 或进入 manual scope。
- 不沿用未在当前安装核实的 TileShape、Stitch、调度参数、lowering 开关或历史阈值。能解析、能
  导出 dtype 或名字相似都不构成使用授权。

## 证据解释

PTOBench evaluate 的 correctness 与 clean latency 针对具体程序和环境；profile 是诊断，不代替
evaluate。`measured`、`derived`、`unavailable` 要分开。缺失不等于零，失败解析不等于没有变化。
比较前确认程序来源、问题、单位、资源集合和测量范围兼容；不兼容的数据不能拼成趋势。

任务数下降、活动周期减少、利用率上升或一次 latency 改善都不单独证明因果。解释时注明证据范围
和未知项；若机制证据不可用，可报告源码变化与 clean latency 结果，但不得把机制说成已验证。
正确性或 clean latency 回退条件触发时，保留失败记录并按 Controller 返回的程序版本处理。
