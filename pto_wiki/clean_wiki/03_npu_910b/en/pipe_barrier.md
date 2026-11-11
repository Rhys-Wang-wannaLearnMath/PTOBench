> **CANN 9.0.X — local 910B4 / CANN 9.0.0 reading edition.** [Official English page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0271.html).

# PipeBarrier(ISASI)

#### Applicability

| Product | Supported |
| --- | --- |
| Atlas 350 Accelerator Card | √ |
| Atlas A3 training product/Atlas A3 inference product | √ |
| Atlas A2 training product/Atlas A2 inference product | √ |
| Atlas 200I/500 A2 inference product | √ |
| Atlas inference product AI Core | √ |
| Atlas inference product Vector Core | x |
| Atlas training product | √ |

#### Function Usage

Blocks a pipeline. This synchronization needs to be inserted between the same pipelines with data dependencies.

#### Prototype
```cpp
template <pipe_t pipe>
__aicore__ inline void PipeBarrier()
```

#### Parameters

| Parameter | Description |
| --- | --- |
| pipe | Template parameter, indicating the type of a blocked pipeline. For details about supported pipelines, see [Pipelines](sync_control_overview.md). If you do not care about the pipeline type and want to block all pipelines, you can pass **PIPE_ALL**. |

#### Returns

None

#### Constraints

The synchronization between Scalar pipelines is automatically ensured by the hardware. Calling **PipeBarrier<PIPE_S>()** throws a hardware error.

#### Example

In the following example, the input **dst0Local** of the Mul instruction is the output of the Add instruction. The two vector operation instructions depend on each other. Therefore, **PipeBarrier** needs to be inserted to ensure the execution sequence of the two instructions.

Note: This is only an example for reference. When automatic synchronization is enabled (enabled by default in kernel launch operator projects and custom operator development projects), the compiler automatically inserts the **PIPE_V** synchronization, and you do not need to manually insert it.
```cpp
AscendC::LocalTensor<half> src0Local;
AscendC::LocalTensor<half> src1Local;
AscendC::LocalTensor<half> src2Local;
AscendC::LocalTensor<half> dst0Local;
AscendC::LocalTensor<half> dst1Local;

AscendC::Add(dst0Local, src0Local, src1Local, 512);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Mul(dst1Local, dst0Local, src2Local, 512);
```
