# PTOBench 双模式运行验收（2026-10-04）

验收目标：保留 `pypto-lib/benchmarks/pypto_lib_bench-v0.2` 的嵌套使用方式，
同时允许将可发布文件作为独立 Git 仓库 clone 并命名为 `PTOBench`。
本轮未修改 Case、Candidate Seed、Fixture、Ref、Oracle 或 comparator。

## 修缮内容

- 统一 benchmark 根目录、外部 PyPTO-Lib 根目录及外置 sealed assets 的解析；
  保留 `PTOBENCH_REPO_ROOT`、`PTOBENCH_SEALED_ROOT` 和三个工具链路径覆盖。
- 29 份环境配置使用路径占位符。嵌套模式保留本机默认配置，独立模式显式配置工具链。
- 保留虚拟环境 Python 的链接路径，避免 re-exec 到基础 Python 后丢失依赖；
  CANN setup script 的主机安装链接在预检中解析到真实文件，canonical 资产仍执行 no-follow 校验。
- 源码树识别同时检查 `benchmarks/` 布局与 `golden/`，独立仓库放在普通
  `benchmarks/` 目录下时不会误用其祖先目录。外部源码的 `golden` 导入不依赖调用目录或继承的 `PYTHONPATH`。
- Pi/OpenCode 保留存在的 PyPTO-Lib provider launcher。仅当内置 provider launcher
  缺失时改用系统 CLI；自定义 wrapper 不被静默丢弃，非法 argv 仍被拒绝。
  系统默认模型记录为 `unknown`，命名 profile 保留指定 provider/model，要求主机配置对应 provider。
- 增加本目录 `.gitignore`，排除 sealed assets、Session、编译输出及大型整批验收产物；
  更新独立运行和测试文档。已有大体量产物保留在本地。
- 增加 4 项 portability 回归；CPU 测试的临时资产不会被操作者的外部 root 设置重定向；
  Comparator 测试使用统一的外置 sealed root，不再写死本地资产目录。

## 控制层测试

嵌套目录完整回归：164 项 / 455.809 秒 / OK，见 [nested_control_tests.log](nested_control_tests.log)。
最终独立配置专项：4 项 / 24.649 秒 / OK，见 [portability_tests.log](portability_tests.log)。
外部路径已配置时的 Session 专项：16 项 / 3.361 秒 / OK，见 [configured_root_session_tests.log](configured_root_session_tests.log)。

独立 Git clone 的最终完整回归：165 项 / 453.602 秒 / OK，见 [standalone_control_tests.log](standalone_control_tests.log) 和 [standalone_control_tests.json](standalone_control_tests.json)。
该回归使用真实 Git init、commit、clone 生成 `PTOBench`，从 `/` 启动，设置外部
源码和 sealed asset 根目录并移除继承的 `PYTHONPATH`。Session/Launcher 等控制用例使用自己创建的小型资产；
Comparator 检查读取 Case evaluator 源码和认证，实际覆盖 195 个有 evaluator 的 Case，
另 3 个未就绪且缺少 evaluator 的 Case 明确记录。这些用例不执行 NPU。

静态解析：2,814 个 Python、1,457 个 JSON、11 个 YAML，无错误，见
[static_validation.json](static_validation.json)。这部分包含 canonical sealed 源码，排除历史运行产物。

## 真实 NPU 检查

代表 Case：`deepseek_v4.rmsnorm_decode_demo`，既有 Fixture 与 Ref cache，未 prepare-ref 或再生成资产。

- 独立 Git clone：从 `/` 启动，移除继承的 `PYTHONPATH`，外挂原 PyPTO-Lib 和 sealed assets。
  clean evaluate、runtime、PMU、compiler 全部通过；三类 native 产物结构校验通过。
  控制驱动正常退出后，无需 submit，Launcher 自动选择历史 Snapshot 并完成独立 clean finish；
  4 次探索 + 1 次 finish，最终 `completed`，status 查询通过。
- 嵌套目录：从 `/` 启动，不设置任何本次路径覆盖，真实 evaluate 和独立 finish 均通过。

见 [standalone_hardware.json](standalone_hardware.json)、[standalone_hardware.log](standalone_hardware.log)、
[nested_hardware.json](nested_hardware.json)、[nested_hardware.log](nested_hardware.log)。
完整日志和诊断目录保留在各 JSON 的 `audit` 路径中。独立硬件验证使用本地受控 CLI 驱动，
评测器与 NPU 执行为真实路径；这项测试不认证远程模型响应或 Agent 登录。
四个实际工具的版本查询均成功，见 [backend_versions.json](backend_versions.json)。

## 发布与使用边界

可发布文件约 25.5 MiB，无超过 100 MiB 的文件；忽略规则实际生效，见
[gitignore_and_profiles.json](gitignore_and_profiles.json)。本轮只创建本地临时 Git 仓库进行验证，没有向 GitHub 发布。
实际 NPU 使用仍需外部 PyPTO-Lib checkout、相应 Python 依赖、CANN/PTOAS 和匹配的 sealed asset bundle；
控制仓库不携带数百 GiB 的受保护二进制及历史 Session。

正常/异常进程退出或探索预算耗尽可以触发自动收尾。进程仍活着但没有输出不等同于结束，
当前没有 inactivity timeout，且预算未配置时默认 20 次探索、1 次 finish。

本轮验收针对双模式部署，没有重跑 198 个 Case 的整批硬件验收。既有
[S7 报告](../agent_upgrade/S7_full_v02_acceptance.md)记录 157 个 Case 四项通过、5 个失败、36 个被阻塞，
不能将控制层回归或代表性硬件检查解读为所有 Case 均通过。

关键文件内容 hash 见 [implementation_manifest.json](implementation_manifest.json)。
