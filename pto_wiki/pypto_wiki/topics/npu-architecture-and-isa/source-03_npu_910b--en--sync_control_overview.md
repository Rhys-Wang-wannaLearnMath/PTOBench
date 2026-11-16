> **CANN 9.0.X — local 910B4 / CANN 9.0.0 reading edition.** Official English page (`https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0179.html`).

# Introduction to Synchronization Control

You can use the synchronization control API to implement synchronization control. Generally, there is no need to consider synchronization when programming is based on the programming model and paradigms described in Programming Model (`https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0012.html`). The programming model implements synchronization control. Using programming model and paradigms is recommended. Manual synchronization control may increase programming complexity, which is not recommended. The difference between the TQueSync (`https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0178.html`) APIs and the synchronization control APIs provided in [SetFlag/WaitFlag(ISASI)](source-03_npu_910b--en--setflag_waitflag.md) is that the APIs in [SetFlag/WaitFlag(ISASI)](source-03_npu_910b--en--setflag_waitflag.md) are of the ISASI type, which cannot guarantee compatibility across hardware versions, while the **TQueSync** APIs can.

#### Overview

Asynchronous parallel computing process of the AI Core: The Scalar Unit reads the instruction sequence and issues the vector computation, matrix computation, and data transfer instructions to the instruction queues of other units. The Vector Unit, Cube Unit, and DMA units asynchronously execute the received instructions in parallel.

Different instructions may depend on each other. To ensure that instructions in different instruction queues are executed based on the correct logic, the Scalar Unit also issues synchronization instructions to other units.

The internal data processing of the AI Core is as follows: The DMA copy-in unit moves data to the local memory. The Vector/Cube Units complete data computation and write the computation result back to the local memory. And then, the DMA copy-out unit moves the processed data back to the global memory.

The execution units in the AI Core are asynchronously parallel. There might be a dependent relationship when the local memory is read or written, requiring synchronization control.

A common vector computation data stream:

- The DMA execution unit transfers data from the global memory to the local memory.

- Computation is performed.

- The DMA execution unit transfers the computation result from the local memory to the global memory.

The four execution units Scalar, Vector, DMA (VECIN), and DMA (VECOUT) are executed in parallel. If the same local memory is accessed, a synchronization mechanism is required to control the access timing. That is, data needs to be moved to the local memory for computation and then moved out after computation.

#### Pipelines

The following describes the parallel instruction pipelines in the AI Core.

The specific pipelines contained in each hardware pipeline type vary according to the hardware architecture. For details, see Hardware Implementation (`https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0007.html`).

| Type | Meaning |
| --- | --- |
| PIPE_S | Scalar computation pipeline, which is used when **Tensor GetValue** is used. |
| PIPE_V | Vector computation pipeline and data transfer pipeline from **L0C Buffer** to **UB** in some hardware architectures. |
| PIPE_M | Matrix computation pipeline. |
| PIPE_MTE1 | Data transfer pipeline from **L1 Buffer** to **L0A Buffer** and from **L1 Buffer** to **L0B Buffer**. |
| PIPE_MTE2 | Data transfer pipeline from **GM** to **L1 Buffer**, from **GM** to **UB**, and more. |
| PIPE_MTE3 | Data transfer pipeline from **UB** to **GM** and more. |
| PIPE_FIX | Data transfer pipeline from **L0C Buffer** to **GM**, from **L0C Buffer** to **L1**, and more. |

#### Synchronization Control Classification

There are two types of parallel pipelines for synchronization control:

- Multi-pipeline synchronization: **SetFlag**/**WaitFlag** of the TQueSync (`https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0178.html`) APIs or the APIs in [SetFlag/WaitFlag(ISASI)](source-03_npu_910b--en--setflag_waitflag.md) are called to control the synchronization between different pipelines.

- **SetFlag**: After all read and write operations of preceding instructions in the source pipeline are complete, SetFlag sets the corresponding hardware flag bit to **1**.

- **WaitFlag**: When this instruction is executed, if the corresponding flag bit is **0**, the subsequent instructions in the queue are blocked; if the corresponding flag bit is **1**, the subsequent instructions are executed after this bit is changed to **0**.

- Single-pipeline synchronization: [PipeBarrier(ISASI)](source-03_npu_910b--en--pipe_barrier.md) is called to implement synchronization control in a single pipeline to restrict the execution sequence. This ensures that all data reads and writes of preceding instructions in the selected pipeline complete before subsequent instructions execute.

#### Requirements on Enabling Manual Synchronization Control

- Vector Unit

- Single-pipeline synchronization: For **PIPE_V**, synchronization is automatically inserted by the compiler. If the movement addresses of **PIPE_MTE2**/**PIPE_MTE3** overlap, you need to insert synchronization. For details, see precautions (`https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0103.html#EN-US_TOPIC_0000002562301699__li71531626153617`).

- Multi-pipeline synchronization: The multi-pipeline synchronization between **PIPE_V**, **PIPE_MTE2**, **PIPE_MTE3**, and **PIPE_S** is bidirectional. These synchronization operations are provided by the compiler and the Ascend C framework. Exception: When an atomic operation API (such as AtomicAdd (`https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00259.html`)) involving scalar computation is used, if the scalar computation unit and the **MTE2**/**MTE3** have data dependencies when reading and writing the global memory, you need to insert synchronization based on the actual situation. For details, see the API description.

- Cube Unit

All pipeline synchronization on the Cube is completed by the Ascend C framework. You do not need to manually insert synchronization.

The synchronization inserted automatically by the compiler strongly depends on local tensors. In some scenarios, you need to manually complete the synchronization. For details about the restrictions, see BiSheng Compiler (`https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/compiler/BishengCompiler/atlas_bisheng_10_0001.html`).

> **PyPTO scope:** The automatic synchronization described above depends on Ascend C LocalTensor/framework/compiler settings. It does not guarantee synchronization for arbitrary PTO ISA or PyPTO extern code; check the selected backend and generated instructions.
