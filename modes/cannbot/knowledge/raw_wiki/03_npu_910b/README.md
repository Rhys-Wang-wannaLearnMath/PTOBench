# NPU 910B Official Source Corpus

官方资料原始版：保留用户指定的四个入口及已选 CANN 9.0.X 英中文历史资料。
本部分不应用本机容量、Tiling 路径或翻译错误等技术修正；两个知识版本目前直接对齐。

英文和中文阅读页由已保存的官方 HTML 正文离线转换，仅整理格式和图片路径。
原始 HTML/Markdown、目录响应和 68 张图片位于 [raw/](raw/)；
指定入口的原文位于 [requested_originals/](requested_originals/)。
来源与版本见 [historical_manifest.json](historical_manifest.json) 和 [fetch_manifest.json](fetch_manifest.json)。

| Topic | English | Chinese |
| --- | --- | --- |
| Basic Architecture | [EN](en/basic_architecture.md) | [ZH](zh/basic_architecture.md) |
| NPU Architecture Version 220x | [EN](en/architecture_220x.md) | [ZH](zh/architecture_220x.md) |
| NPU Architecture Version 220x | [EN](en/constraints_220x.md) | [ZH](zh/constraints_220x.md) |
| Overview | [EN](en/debugging_overview.md) | [ZH](zh/debugging_overview.md) |
| CPU Twin Debugging | [EN](en/cpu_twin_debugging.md) | [ZH](zh/cpu_twin_debugging.md) |
| Board Debugging on the NPU | [EN](en/npu_debugging.md) | [ZH](zh/npu_debugging.md) |
| Avoiding Bank Conflicts (NPU Architecture Version 220x) | [EN](en/bank_conflicts_220x.md) | [ZH](zh/bank_conflicts_220x.md) |
| How to Use the Header File on Which Tiling Depends | [EN](en/tiling_headers.md) | [ZH](zh/tiling_headers.md) |
| SetFlag/WaitFlag(ISASI) | [EN](en/setflag_waitflag.md) | [ZH](zh/setflag_waitflag.md) |
| PipeBarrier(ISASI) | [EN](en/pipe_barrier.md) | [ZH](zh/pipe_barrier.md) |
| CrossCoreSetFlag (ISASI) | [EN](en/cross_core_setflag.md) | [ZH](zh/cross_core_setflag.md) |
| CrossCoreWaitFlag (ISASI) | [EN](en/cross_core_waitflag.md) | [ZH](zh/cross_core_waitflag.md) |
| AllocEventID | [EN](en/alloc_event_id.md) | [ZH](zh/alloc_event_id.md) |
| FetchEventID | [EN](en/fetch_event_id.md) | [ZH](zh/fetch_event_id.md) |
| Introduction to Synchronization Control | [EN](en/sync_control_overview.md) | [ZH](zh/sync_control_overview.md) |
