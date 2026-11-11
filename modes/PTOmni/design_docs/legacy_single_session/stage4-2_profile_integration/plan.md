# Stage 4-2：Profile 事实接入在线运行

## 目的

让 Pi 能针对 Hypothesis 请求 Profiling Tool，并把完整、客观、版本绑定的观测写入唯一 Progress State。

## 依赖

- Stage 0 的 Observation/EvidenceBundle 契约；
- Stage 1-2 完成 PTOBench 原始 Profile 的读取和呈现整理；
- Stage 2 的 Progress State；
- Stage 3 的 ContextRenderer 和 Pi 持续循环。

本阶段可与 `stage4-1_knowledge_integration` 并行。

## 实现内容

1. Controller 只接受绑定当前 Session、Hypothesis 和 ProgramRef 的 Profile 请求。
2. 通过 PTOBench 的公开 Profile 入口取得原始工件，再交给 Profiling Tool 整理。
3. 将完整结构化观测关联到 Cycle 和 ProgramRef；按问题返回的短视图不能删除完整事实。
4. Progress Reducer 只登记事实、字段可用性和引用；Pi 的解释单独保存为提议。
5. Profile 失败、部分缺失、未请求和结果无变化都能被明确表示。

## 验收

- 任何返回的指标均可追溯到受控 Profile 结果和准确程序版本。
- 原始与结构化 Profile 均未被信息筛选破坏；Agent 不会收到主观瓶颈/因果结论。
- Profile 预算计入 PTOBench 的公开预算。
- Profile 查询失败不会覆盖之前的有效 EvidenceBundle。
