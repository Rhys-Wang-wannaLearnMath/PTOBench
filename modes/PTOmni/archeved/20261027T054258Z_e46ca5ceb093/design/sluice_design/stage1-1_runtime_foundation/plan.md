# Stage 1-1：运行基础和 PTOBench 适配

## 目的

建立最小的单 Pi 运行骨架：一个 Session、一个 Pi bridge、一个 Controller 和公开 PTOBench 动作入口。这个阶段不实现完整优化策略。

## 实现内容

1. Launcher 创建一次 PTOBench Session 和一次 Pi 进程。
2. Bridge 注册 SLUICE 的最小工具入口，将调用交给 Controller。
3. 使用 `benchmarks/PTOBench/modes/_shared/profiles/pi/deepseek-flash-max.yaml` 启动 Pi；本地环境以 `benchmarks/PTOBench/modes/basic/.env` 为模板复制到未纳入版本控制的运行目录，绝不将密钥写入仓库。
4. PTOBench Adapter 封装实际存在的公开 workspace、evaluate、profile、预算和收尾接口；不猜测或复制内部接口。
5. Controller 统一检查 Session、ProgramRef、预算、调用顺序和状态 revision。
6. 将 PTOBench 原始结果归一化为 Stage 0 的公共结果结构。
7. 统一处理编译失败、correctness 失败、预算拒绝、Session 关闭和 Adapter 异常。

## 接口原则

- Caller 只提交一个动作请求，不直接拼接 PTOBench 内部参数。
- Adapter 隐藏路径、进程和结果解析细节。
- 不创建第二个 Pi、第二个 Controller 或临时状态库。
- Profile 和 Knowledge 先以可替换的只读端口接入，不能在 Controller 中复制它们的逻辑。

## 验收

- 一个 Session 只启动一个 Pi。
- 非法 Session、错误 ProgramRef、过期 revision 和超预算动作会被拒绝。
- evaluate 结果能区分成功、失败、无提升和不可用。
- 所有失败路径不会把未验证程序写成 `best_verified`。

## 不做

- 不实现机会排序、speedup 预测或自动因果判断。
- 不实现恢复运行。
- 不把 Pi 的普通对话历史交给 Controller 管理。
