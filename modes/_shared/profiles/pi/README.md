# Pi profile token 配置

```yaml
context_window: 1048576
max_output_tokens: 393216  # DeepSeek Flash 官方最大值；GLM 使用 131072
```

`context_window` 为正整数，`max_output_tokens` 为正整数或 `unlimited`。
修改 profile 对后续启动的实验生效，不修改已启动 Run 的冻结配置。
配置通过 mode-local `.env` 指定的 endpoint/API 写入各 Run 独立的 Pi 配置；
不修改共享 `.env`、宿主配置或 Pi 安装。

截至 2026-10-19，五个 YAML 默认显式请求各模型官方最大输出，避免不传参数时使用较小服务端默认值：

| profiles | 模型 | 官方最大上下文 | context_window | max_output_tokens |
| --- | --- | --- | --- | --- |
| deepseek-flash、deepseek-flash-max | deepseek-flash | 1M | 1048576 | 393216 |
| default、glm-5.3-flash、glm-5.3-flash-max | glm-5.3-flash | 1M | 1048576 | 131072 |

官方页面使用“1M”标记，本仓库按本轮要求采用 1,048,576 tokens。
来源：[DeepSeek 官方规格](https://api-docs.deepseek.com/quick_start/pricing)、
[GLM-5.3-Flash 官方规格](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash)。
官方输出上限分别为 384K（393216）、128K（131072）。
DeepSeek [Chat API 参数说明](https://api-docs.deepseek.com/api/create-chat-completion) 明确：
未设置上限时，非思考为 8K、思考为 64K、`max` 为 128K；不等于官方最大输出。
Responses 参数页未明确承诺默认取最大，因此本轮按用户兜底要求显式设置官方最大值。
`unlimited` 仍受支持，表示省略客户端 cap，由服务端默认值决定；它不保证使用最大值。
原生 Pi 可能根据剩余上下文缩小有限输出预算，输入加输出不能突破总上下文。
`gpt-6.1-sol.yaml` 已按用户要求删除。
