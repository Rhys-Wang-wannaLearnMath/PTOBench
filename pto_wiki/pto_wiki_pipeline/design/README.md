# PTO Wiki 整理 Agent Workflow

> 状态：已实现，入口见本目录的 `README.md` 与 `workflow.py`。

## 目标

把 `clean_wiki` 整理成自包含的 `pypto_wiki`，供 PTOmni 直接使用。只做保守整理：集中相关资料、合并明确重复内容、提供简短索引；不大幅改写知识内容。

`clean_wiki` 是只读来源归档。PTOmni 运行时只接入 `pypto_wiki`。

## 输入与输出

```text
clean_wiki/INPUT_MANIFEST.json + 清单列出的文件 + 整理规则
    → pypto_wiki + provenance/<run-id>/ + .work/<run-id>/
```

- 输入清单是唯一来源范围。每次运行开始及阶段切换时校验路径、文件大小和 SHA-256；有变化就停止，不自行更新清单。
- 不读取清单以外的 wiki 文件、归档、Case、Fixture、Ref、Oracle 或实验结果。
- `pypto_wiki` 包含简短 `README.md`、主题索引 `INDEX.md` 和整理后的主题文档，脱离 `clean_wiki` 仍可使用。
- `provenance/<run-id>/` 保存输入到输出的来源映射、独立核查结果和运行参数，不保存 API key。
- `.work/<run-id>/` 保存候选知识库及中间响应，仅供维护者检查，不提供给 PTOmni。

## 三个阶段

1. **Mapping：** Agent 按使用者会查找的问题组织少量主题，标记明确重复和未确定的重叠。映射需覆盖全部输入，不修改来源。
2. **Organization：** Agent 将资料整理到候选 `pypto_wiki`。语境相同的明确重复内容可以合并；不同版本、硬件或上下文保持区分。保留原有解释、代码、表格和术语，只调整导航、标题和必要链接。不添加优化建议、优先级、风险判断、性能结论或来源之外的新陈述。
3. **Independent review：** 独立 Agent 对照清单、来源和映射，只报告问题。整理 Agent 最多按核查结果修订两次；全部通过后才发布候选版本。

各阶段使用独立 Codex 调用。Codex 执行沙箱关闭网络访问，模型 API 请求由 Codex CLI 发起。整理流程不调用 PTOBench、不运行 NPU，也不属于 PTOmni 的优化轮次。

## 保护与发布

- `clean_wiki` 只读；manifest 源文件哈希不匹配时中止。
- 映射与复核使用只读沙箱；整理 Agent 只写候选目录。
- 本地检查确保所有输入都映射到输出、输出路径留在候选内、必要文件存在、内部 Markdown 链接有效。
- 候选版本须通过独立核查后才替换 `pypto_wiki`。发布前保留旧版到该运行目录，便于恢复。
- API key 只从本地忽略文件 `.env` 读取，不写入产物或运行元数据。

## 配置

`pto_wiki_pipeline/.env.example` 是无密钥模板。本地 `.env` 已预设 `codex`、`gpt-6.1-sol`、`high` 和 `https://api.aixuanhub.me/v1`，只需填入 `PTO_WIKI_API_KEY`。`.env` 不纳入版本控制。

## 验收

- 输入严格来自 manifest；输入变化会停止。
- 映射与来源记录覆盖每个输入文件。
- 整理内容可独立阅读，索引与内部链接有效。
- 独立核查通过后才发布，旧版可从运行目录恢复。
