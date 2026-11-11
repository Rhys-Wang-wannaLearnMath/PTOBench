> **CANN 9.0.X — local 910B4 / CANN 9.0.0 reading edition.** [Official English page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0008.html).

# Basic Architecture

> **Local 910B4:** A2 / NpuArch 2201, separated AIC/AIV, 20 Cube cores and 40 Vector cores per device; 32 GiB HBM and 96 MiB L2. The installed UB capacity is 192 KiB per AIV; pinned PyPTO caps the Vec allocation budget at 184 KiB before any additional kernel reservations. See the [environment baseline](../../BASELINE.md).

Operators developed using Ascend C run on the AI Core. The Ascend C programming model is introduced based on the abstraction of the AI Core hardware architecture. Understanding the hardware architecture helps you better understand the programming model. For experienced developers who need to complete high-performance programming, it is even more important to learn about the hardware architecture. Many contents in the operator practice reference are introduced based on this chapter.

The AI Core is responsible for executing cube and vector computation–intensive tasks. It consists of the following components:

- **Compute units**: include the Cube Unit, Vector Unit, and Scalar Unit.

- **Storage units**: include the L1 Buffer, L0A Buffer, L0B Buffer, L0C Buffer, Unified Buffer, BiasTable Buffer, and FixPipe Buffer, which are designed for efficient computation.

- **Movement units**: include the MTE1, MTE2, MTE3, and FixPipe, which are used for efficient data transmission between different storage units.

This section first introduces the key concepts and terms related to the hardware architecture and the working mode of the AI Core, laying a foundation for understanding the subsequent content. Then, using  Atlas A2 training product / Atlas A2 inference product  as an example, this section provides an introduction to the basic architecture of the AI Core. It first describes the basic functions and structures of the compute units, storage units, and movement units, and then uses typical data flow and control flow examples to help you gain a deeper understanding of the working principles of the hardware architecture. For details about the specific architecture specifications of different product models, see [Architecture Specifications](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0009.html).

#### Key Concepts and Terms

- Core

Compute core with an independent Scalar Unit. The Scalar Unit is a scheduling unit within a core and is responsible for functions such as instruction transmission within the core.

- AI Core

A compute core of an AI processor, which executes tasks with intensive cube and vector compute.

- Cube Core

A core dedicated for cube computation. It consists of the Scalar Unit, Cube Unit, and DMA Unit, but does not include the Vector Unit.

- Vector Core

A core dedicated for vector computation. It consists of the Scalar Unit, Vector Unit, and DMA Unit, but does not include the Cube Unit.

- AIC

A Cube Core in a group of Cube Cores and Vector Cores in [separated mode](basic_architecture.md) of AI Core.

- AIV

A Vector Core in a group of Cube Cores and Vector Cores in [separated mode](basic_architecture.md) of AI Core.

#### AI Core Working Modes

- Separated mode

A working mode of AI Core. The Cube Unit and Vector Unit each have their own Scalar Unit, and they are deployed on the Cube Core and Vector Core, respectively. By combining Cube Cores and Vector Cores in a certain ratio (1:N), such a combination is considered as an AI Core. The number of AI Cores is determined by the number of Cube Cores.

- Coupled mode

A working mode of AI Core. The Cube Unit and Vector Unit share the same Scalar Unit, which is deployed on an AI Core.

 In Ascend C programming, the working modes of different products are as follows:

- Atlas inference product : coupled mode

- Atlas training product : coupled mode

- Atlas A2 training product / Atlas A2 inference product : separated mode

- Atlas A3 training product / Atlas A3 inference product : separated mode

- Atlas 350 Accelerator Card: separated mode

- Atlas 200I/500 A2 inference product : coupled mode

Note: For the  Atlas 200I/500 A2 inference product , the hardware working mode can be either coupled mode or separated mode. In coupled mode, you only need to concern about the number of AI Cores, and do not need to concern about the number of Vector Cores and Cube Cores. In separated mode, you need to concern about the number of AI Cores, Vector Cores, and Cube Cores. In Ascend C programming scenarios, only the coupled mode is supported.

#### Compute Units

The compute units are the core of powerful compute capabilities of the AI Core, including the Cube Unit, Vector Unit, and Scalar Unit. These three units process different types of data in the AI Core.

- Cube

The Cube Unit performs cube operations. Take the float16 data type as an example. The Cube Unit can multiply two 16×16 cubes of the float16 data type in each execution. L0A stores the left cube, L0B stores the right cube, and L0C stores the cube multiplication result and intermediate result.

- Vector

The Vector Unit performs vector computation. It executes vector instructions, which are similar to conventional single instruction multiple data (SIMD) instructions. Each vector instruction can complete the same type of operation for a plurality of operands. The Vector Unit can quickly add or multiply two float16 vectors. Vector instructions support multiple repeats and direct computation of vectors with intervals.

All the computed source data and target data of the vector must be stored in the Unified Buffer. The start address and operation length of vector instructions have alignment requirements, typically 32 bytes. For details, see the API constraints.

- Scalar

The Scalar Unit performs scalar computation and controls the program flow. It can be regarded as a mini CPU, which implements loop control, branch judgment, address and parameter computations of Cube and Vector instructions, and basic arithmetic operations for a program. In addition, the pipeline of other execution units in the AI Core can be controlled by inserting a synchronization code into the event synchronization module. Compared with the host CPU, the scalar computation capability on the AI Core is relatively weak, and scalars are mainly used to send instructions. Therefore, scalar computations should be minimized in actual application scenarios, for example, minimizing the branch judgment such as the **if/else** statement and variable computation during performance optimization.

The Scalar Unit executes standard arithmetic logic unit (ALU) statements when executing scalar operation instructions. The code segment and data segment (stack space) required by ALU come from the GM. The instruction cache (iCache) is used to cache code segments, and the cache size is related to hardware specifications. If the size is 16 KB or 32 KB, the code segment is loaded in the unit of 2 KB. The data cache (dCache) is used to cache data segments, and the cache size is related to hardware specifications. If the size is 16 KB, the data segment is loaded in the unit of cache lines (64 bytes). Considering that access inside the core is most efficient, ensure that code segments and data segments are cached in the iCache and dCache to avoid access outside the core. In addition, based on the data loading unit, you can adjust the size of data to be loaded at a time during programming to improve the loading efficiency. For example, when data is loaded to the dCache and the start address of the data memory is aligned with the cache lines (64 bytes), the loading efficiency is the highest.

The hardware provides the L2 cache to cache the data (including code segments and data segments) that accesses the GM, thereby accelerating the access speed and improving the access efficiency. The L2 cache outside the core loads data in the unit of cache lines. The cache line size (128, 256, or 512 bytes) varies with hardware specifications.

#### Storage Units and DMA Units

To fully leverage the powerful computing capability of the AI processors, the prerequisite is that the input data can be fed to the compute units in a timely and accurate manner. Therefore, the memory system needs to be elaborately designed to ensure the data supply required by the compute units.

An AI Core includes multiple levels of local memory (internal storage), and loads data from the global memory (external storage) to the local memory for computation. The local memory of an AI Core includes the L1 Buffer, L0 Buffer, Unified Buffer, and so on. An AI Core also contains memory transfer engines (MTEs) to facilitate data movement and copy, which can convert the data format or type during data movement.

The following tables describe the internal storage units and DMA units.

| Storage Unit | Description |
| --- | --- |
| L1 Buffer | As a general part of the local memory, the L1 Buffer offers a large data buffer in the AI Core. It can temporarily store data that needs to be repeatedly used in the Cube Unit, thereby reducing the frequency that data is read from or written to the bus. |
| L0A Buffer/L0B Buffer | They store the inputs of Cube instructions. |
| L0C Buffer | It stores the outputs of Cube instructions, which are part of the inputs for accumulation. |
| Unified Buffer | The Unified Buffer stores the inputs and outputs of vector and scalar computations. |
| BT Buffer | The BiasTable (BT) Buffer stores the bias in cube computations. |
| FP Buffer | The FixPipe (FP) Buffer stores quantization parameters and ReLU parameters. |

| DMA Unit | Description |
| --- | --- |
| MTE1 | Moves data in the following paths: L1->L0A/L0B; L1->BT Buffer |
| MTE2 | Moves data in the following paths: GM->{L1, L0A/B}. In this path, data is moved based on the fractal size. If the data movement meets the cache line size alignment requirements, the performance is better. GM->UB. Data movement based on the cache line size will achieve better performance. |
| MTE3 | Moves data in the following paths: UB -> GM |
| FixPipe | Moves data in the following paths. (The data format or type can be converted during data movement.); L0C->{GM/L1}; L1->FP Buffer |

- The size of a storage unit varies with the AI processor type. You can call [GetCoreMemSize](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1034.html) to get the size.

- By default, all data that is read from or written to the GM through the DMA units is cached in the L2 cache to accelerate the access speed and improve the access efficiency. The L2 cache outside the core loads data in the unit of cache lines. The cache line size (128, 256, or 512 bytes) varies with hardware specifications.

#### Typical Data Flows

- Typical data flow of vector computation:

GM → UB → Vector → UB → GM

- Typical data flows of cube computation:

- GM →L1→L0A/L0B →Cube →L0C→FixPipe→GM

- GM →L1→L0A/L0B →Cube →L0C→FixPipe→L1

#### Typical Instruction Flows

Multiple instructions enter the iCache from the system memory through the bus. Based on the instruction type, there are two kinds of subsequent instruction execution processes:

- For a scalar instruction, it will be executed immediately by the Scalar Unit.

- For other instructions, they are scheduled to independent queues (vector instruction queue, cube instruction queue, and MTE1/MTE2/MTE3 instruction queues) by the Scalar Unit, and then executed by corresponding execution units.

 Instructions in the same queue are executed according to their enqueue sequence, and instructions in different queues can be executed in parallel. Such parallelism improves overall execution efficiency. For data dependency that may occur during parallel execution, the Event Sync module inserts synchronization instructions to control pipeline synchronization. The **PipeBarrier** and **SetFlag**/**WaitFlag** APIs are provided to ensure that the instructions in a queue and across queues are executed based on the logical relationship.

- **PipeBarrier** is an instruction used to control the execution sequence in a queue. (Although instructions are executed in sequence, it does not mean that the execution of the previous instruction ends when the next instruction starts to be executed.) **PipeBarrier** ensures that all data read and write operations of the previous instruction are completed before the next instruction is executed.

- **SetFlag** and **WaitFlag** are a pair of inter-queue synchronization instructions.

- **SetFlag**: After all read and write operations of preceding instructions in the source pipeline are complete, SetFlag sets the corresponding hardware flag bit to **1**.

- **WaitFlag**: When this instruction is executed, if the corresponding flag bit is **0**, the subsequent instructions in the queue are blocked; if the corresponding flag bit is **1**, the subsequent instructions are executed after this bit is changed to **0**.

Ascend C provides [synchronization control APIs](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00013.html). You can use this type of APIs to implement synchronization control. Generally, there is no need to consider synchronization when programming based on the programming model and paradigm described in [Programming Model](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0012.html). The programming model implements synchronization control. Using the programming model and paradigm is recommended. Manual synchronization control may complicate programming.

However, we still hope that you can understand the basic principles of synchronization to better understand and design parallel compute programs. In a few cases, you need to manually insert synchronization. For details, see [Requirements on Enabling Manual Synchronization Control](sync_control_overview.md).

> **PyPTO scope:** The automatic synchronization described above depends on Ascend C LocalTensor/framework/compiler settings. It does not guarantee synchronization for arbitrary PTO ISA or PyPTO extern code; check the selected backend and generated instructions.
