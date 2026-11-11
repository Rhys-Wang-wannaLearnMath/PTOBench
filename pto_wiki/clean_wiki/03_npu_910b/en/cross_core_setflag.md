> **CANN 9.0.X — local 910B4 / CANN 9.0.0 reading edition.** [Official English page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0273.html).

# CrossCoreSetFlag (ISASI)

#### Applicability

| Product | Supported |
| --- | --- |
| Atlas 350 Accelerator Card | √ |
| Atlas A3 training product / Atlas A3 inference product | √ |
| Atlas A2 training product / Atlas A2 inference product | √ |
| Atlas 200I/500 A2 inference product | x |
| Atlas inference product AI Core | x |
| Atlas inference product Vector Core | x |
| Atlas training product | x |

#### Function Usage

Sets the inter-core synchronization control for the separate mode.

This API works with [CrossCoreWaitFlag](cross_core_waitflag.md). The inter-core synchronization flag ID (**flagId**) needs to be passed. Each ID corresponds to a counter for controlling synchronization.

The synchronization control has the following modes:

- Mode 0: synchronization control between AI Cores. In the AIC scenario, all AIC cores are synchronized. The subsequent instructions of **CrossCoreWaitFlag** are executed only when all AIC cores execute **CrossCoreSetFlag**. In the AIV scenario, all AIV cores are synchronized. The subsequent instructions of **CrossCoreWaitFlag** are executed only when all AIV cores execute **CrossCoreSetFlag**.

- Mode 1: synchronization control between AIV cores in the AI Core. The subsequent instructions of **CrossCoreWaitFlag** are executed only when both AIV cores run **CrossCoreSetFlag**.

- Mode 2: synchronization control between AIC and AIV cores in the AI Core. The subsequent instructions of **CrossCoreWaitFlag** on the two AIV cores are executed only after **CrossCoreSetFlag** is executed on the AIC core, and vice versa.

- Mode 4: synchronization control between AIC and AIV cores in the AI Core. AIV0 and AIV1 can trigger AIC wait independently. For example, the subsequent instructions of **CrossCoreWaitFlag** on AIV0 are executed only after the AIC core executes **CrossCoreSetFlag**, and vice versa.

Mode 4 is supported only by the Atlas 350 Accelerator Card.

#### Prototype
```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
```

#### Parameters

| Parameter | Description |
| --- | --- |
| modeId | Inter-core synchronization mode. The options are as follows: Mode 0: synchronization control between AI Cores. Mode 1: synchronization control between AIV cores in the AI Core. Mode 2: synchronization control between AIC and AIV cores in the AI Core. Mode 4: synchronization control between AIC and AIV cores in the AI Core. AIV0 and AIV1 can trigger AIC wait independently. |
| pipe | The pipeline type of the instruction. For details about the pipeline types, see [Pipelines](sync_control_overview.md). Note that the **PIPE_S** type is supported only by the Atlas 350 Accelerator Card. |

| Parameter | Input/Output | Description |
| --- | --- | --- |
| flagId | Input | Flag ID of the inter-core synchronization. For the Atlas A2 training product / Atlas A2 inference product , the value ranges from 0 to 10. For the Atlas A3 training product / Atlas A3 inference product , the value ranges from 0 to 10. For the Atlas 350 Accelerator Card, the value ranges are as follows: The operations with flag IDs 0 to 10 in **CrossCoreSetFlag** initiated by AIV0 correspond to the operations with flag IDs 0 to 10 in **CrossCoreWaitFlag** on the AIC. The operations with flag IDs 0 to 10 in **CrossCoreSetFlag** initiated by AIV1 correspond to the operations with flag IDs 16 to 26 in **CrossCoreWaitFlag** on the AIC. The operations with flag IDs 0 to 10 in **CrossCoreSetFlag** initiated by AIC correspond to the operations with flag IDs 0 to 10 in **CrossCoreWaitFlag** on AIV0. The operations with flag IDs 16 to 26 in **CrossCoreSetFlag** initiated by AIC correspond to the operations with flag IDs 0 to 10 in **CrossCoreWaitFlag** on AIV1. |

#### Returns

None

#### Constraints

- When using this synchronization API, you need to [set the kernel type](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0218.html) according to the following rules:

- In pure vector or cube scenarios, set the kernel type to **KERNEL_TYPE_MIX_AIV_1_0** or **KERNEL_TYPE_MIX_AIC_1_0**.

- In vector and cube fused scenarios, set the kernel type based on the actual situation.

- This API is used to control inter-core synchronization in the internal implementation of the [Matmul high-level API](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0612.html). Therefore, you are not advised to use this API and the Matmul high-level API at the same time. Otherwise, flag IDs may conflict.

- The counter of the same **flagId** can be set for a maximum of 15 times.

- When using mode 0 of this API, you are advised to enable the batch mode so that an operator exclusively occupies all required core resources. Otherwise, a deadlock may occur if the following conditions are met:

- Multi-stream concurrency scenarios (≥ 2 execution streams)

- Concurrent execution of at least two operators.

- The total number of cores of all concurrent operators exceeds the number of physical cores.

- Two or more concurrent operators use inter-core synchronization.

Specifically, in a multi-stream scenario, although n physical cores are allocated to an inter-core synchronization operator of a stream, only (n – m) cores may be scheduled and executed first, and remaining m cores are not started because they are preempted by inter-core synchronization operators of other streams. When the first started (n – m) cores reach inter-core synchronization, they wait for the remaining m cores to complete execution. However, the remaining m cores are occupied by the inter-core synchronization operators of other streams and cannot be released, resulting in a deadlock.

In kernel direct debugging scenarios, use the [__schedmode__(mode)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10053.html#EN-US_TOPIC_0000002531362270__li1365012910475) qualifier to set the batch mode. In project-based operator development scenarios, use the **SetScheduleMode** API of TilingContext to set the batch mode. For details, see [Basic Data Structures and API List](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/maintenref/basicdataapi/atlasopapi_07_00001.html).

#### Example
```cpp
// Synchronize all AIV cores in mode 0.
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag(0x8);
}

// Synchronize all AIV cores in the AI Core in mode 1.
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag(0x8);
}

// Note that if the high-level API is called, you do not need to process the synchronization between the AIC and AIV cores.
// Take Matmul as an example. The AIC core instructs the AIV to perform subsequent processing after completing the Matmul computation.
if (g_coreType == AscendC::AIC) {
    // Matmul computation
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
}

// The AIV waits for the AIC Set message and performs subsequent processing.
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreWaitFlag(0x8);
    // Subsequent AIV processing.
}
```
