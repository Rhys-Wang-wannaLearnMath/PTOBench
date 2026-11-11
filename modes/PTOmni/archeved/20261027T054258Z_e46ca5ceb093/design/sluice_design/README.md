# SLUICE 开发计划目录

本目录只维护 SLUICE 的当前架构、实现阶段和进展记录。

- [`architecture_v0.md`](architecture_v0.md)：当前初步架构和职责边界；实施中允许按真实情况删改。
- [`implementation_plan.md`](implementation_plan.md)：阶段、依赖和并行关系。
- [`progress.md`](progress.md)：整体进展记录。
- `stage0_contracts/`：先固定共享概念和数据契约。
- `stage1-1_runtime_foundation/`：单 Pi、PTOBench 适配和 Controller 骨架。
- `stage1-2_profile_tool/`：PTOBench 原始 Profile 的读取、呈现和接入契约。
- `stage1-3_knowledge_base/`：离线知识库重新生成工作流和发布格式。
- `stage2_state_reducer/`：唯一 Progress State、事件和确定性 Reducer；可与 stage1 并行。
- `stage3_context_pi_loop/`：每请求 Context 投影和单 Pi 持续运行。
- `stage4-1_knowledge_integration/`、`stage4-2_profile_integration/`：两个只读输入端口，可并行接入。
- `stage5_opportunity_decision/`：在唯一状态中实现有证据依据的下一步决策。
- `stage6_stop_acceptance/`：停止控制、端到端验收和冗余清理。

阶段名称中的 `1-1`、`1-2`、`1-3` 和 `4-1`、`4-2` 表示它们在依赖满足后可以并行推进；不是连续的小版本。每个阶段的 `plan.md` 可作为 Codex Goal 的任务说明，完成后更新同目录 `progress.md` 和总 `progress.md`。
