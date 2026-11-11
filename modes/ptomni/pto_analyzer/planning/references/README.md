# PTOAnalyzer 参考材料

这里保留三个参考项目中与 PTOAnalyzer 九个 Micro-tool 直接相关的源码和说明，供阅读、比较设计使用；它们不是 PTOAnalyzer 的运行依赖，也没有被接入其实现。

| 项目 | 固定版本 | 保留内容 | 许可与使用边界 |
|---|---|---|---|
| [PyPTO-Agent](https://gitcode.com/weixin_70386148/Pypto-Agent) | `7d1197a1fc4e8af13c2c6e38de2df4e98882c171` | `projects/pypto-agent/engine/diagnosis/`、`engine/report/compactness.py`、`LICENSE` | 保留仓库许可证（CANN Open Software License Agreement 2.0）。其 AIC/AIV、bubble、运行事件和源码关联语义仍需按 PTOBench 数据契约重新核实。 |
| [PyPTO-Gym](https://gitcode.com/cann/pypto-gym) | `3c9734211bda5433c868f6d83c92f8d81261f23d` | 与性能分析相关的 CANNBot 主 Skill、`perf-analyzer`、`tune-swimlane` 脚本及少量对应参考文档；另保留 `LICENSE` | 保留仓库许可证（默认适用 CANN Open Software License Agreement 2.0；本次未保留另行许可的 Transformers 组件）。脚本假定其自身的采集格式和硬件语义，不能直接当作 PTOBench 的指标契约。 |
| [awesome-ascend-skills](https://github.com/ascend-ai-coding/awesome-ascend-skills) | `75ec89ce9d401be8ac95980e1c3f42f8d4315057` | 旧同步镜像中的 CANNBot 主 Skill、`perf-analyzer` 和 `tune-swimlane` 说明 | 快照未发现仓库级许可证。只作旧镜像版本比对，不直接复用其中代码；复用前需确认上游版本和许可。 |

解压后的目录按来源隔离：

- `projects/pypto-agent/`：症状、定位、源码/编译器归因及 gap 汇总实现。
- `projects/pypto-gym/`：当前上游的性能分析与 swimlane 工具、说明和有限的优化背景材料。
- `projects/awesome-ascend-skills/`：用于和当前上游核对的旧镜像材料。

下载的完整 ZIP 已解压后删除；只保留上表列出的文件，没有复制仓库 `.git`、大型 Fixture、Case 数据或无关项目内容。来源快照的 SHA-256（用于记录当时下载的归档版本）：

```text
PyPTO-Agent:          6807643bff548963918f5a424294b7a7a0888ea01f0fc7341affac0b8deb61e8
PyPTO-Gym:            f3c1a0460509fe9975aa90d9a8ea2361c29195e05e8fa04900e70d5ecda25900
awesome-ascend-skills: 9f17b3550c83ed472ba33b21b962de126495dfa31ba65320483a29a510f26272
```

九个 Tool 的逐项分析和边界见 [`ANALYSIS.md`](ANALYSIS.md)。所有建议都以 PTOBench 已归档数据的字段、单位、身份和采集覆盖为准；外部项目的公式、阈值和根因判断不能直接移植。
