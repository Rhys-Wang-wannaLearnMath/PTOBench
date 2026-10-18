> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../zh/npu_debugging.md); [official page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0075.html); [unaltered source body](../raw/6ef273448c98b9bc.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="EN-US_TOPIC_0000002531522168"></a><a name="EN-US_TOPIC_0000002531522168"></a> 

# Board Debugging on the NPU

Board debugging methods in the NPU domain include board data printing, msSanitizer memory exception detection, and msDebug single-step debugging. Data printing can be performed using **printf** or **DumpTensor**. **DumpTensor** is a unique function of SIMD programming and is used to print data of a specified tensor.

 <a name="EN-US_TOPIC_0000002531522168__section166162435313"></a><a name="section166162435313"></a> 

#### Using printf to Print Data

**printf** is used to print scalar and character string information and is supported by both SIMT and SIMD programming.

The following is an example of printing data using **printf**. For details about the usage and restrictions of the **printf** API, see [printf](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_10426.html).

```cpp
printf("fmt string %d", 0x123);
```

![](../raw/95ff0a306791d19e.png)

Printing using the **printf** API affect the actual runtime performance of the operator. Therefore, this function is usually used in the debugging phase. You can disable the printing function as required. For details, see [printf](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_10426.html).

#### Debugging SIMD Programming with DumpTensor Printing

**DumpTensor** is a unique printing function in SIMD programming scenarios. It is used to print the data of a specified tensor on the NPU.

**The specific usage is as follows:**

Call the **DumpTensor** API to print required log information at the target position in the operator kernel implementation code.

 In the following example, **srcLocal** indicates the tensor to be printed, **5** indicates the additional custom information, such as the current code line number, and **dataLen** indicates the number of elements. For details about the usage and restrictions of the **DumpTensor** API, see [DumpTensor](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0192.html). 

```cpp
DumpTensor(srcLocal,5, dataLen);
```

During dump, the corresponding **DumpHead** (32 bytes) is added before the dump information of each block core to record the core ID and resource usage. **DumpTensorHead** (32 bytes) is also added before the tensor data to be dumped each time to record tensor information. An example of the printing result is as follows:

```cpp
DumpTensor: desc=5, addr=0, data_type=float16, position=UB, dump_size=32
[19.000000, 4.000000, 38.000000, 50.000000, 39.000000, 67.000000, 84.000000, 98.000000, 21.000000, 36.000000, 18.000000, 46.000000, 10.000000, 92.000000, 26.000000, 38.000000, 39.000000, 9.000000, 82.000000, 37.000000, 35.000000, 65.000000, 97.000000, 59.000000, 89.000000, 63.000000, 70.000000, 57.000000, 35.000000, 3.000000, 16.000000,
42.000000]
DumpTensor: desc=5, addr=100, data_type=float16, position=UB, dump_size=32
[6.000000, 34.000000, 52.000000, 38.000000, 73.000000, 38.000000, 35.000000, 14.000000, 67.000000, 62.000000, 30.000000, 49.000000, 86.000000, 37.000000, 84.000000, 18.000000, 38.000000, 18.000000, 44.000000, 21.000000, 86.000000, 99.000000, 13.000000, 79.000000, 84.000000, 9.000000, 48.000000, 74.000000, 52.000000, 99.000000, 80.000000,
53.000000]
...
DumpTensor: desc=5, addr=0, data_type=float16, position=UB, dump_size=32
[35.000000, 41.000000, 41.000000, 22.000000, 84.000000, 49.000000, 60.000000, 0.000000, 90.000000, 14.000000, 67.000000, 80.000000, 16.000000, 46.000000, 16.000000, 83.000000, 6.000000, 70.000000, 97.000000, 28.000000, 97.000000, 62.000000, 80.000000, 22.000000, 53.000000, 37.000000, 23.000000, 58.000000, 65.000000, 28.000000, 4.000000,
29.000000]
```

![](../raw/95ff0a306791d19e.png)

The print function of the **DumpTensor** API impacts operator runtime performance, so it is generally used for debugging. You can disable the printing function as required. For details, see [DumpTensor](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0192.html).

#### Using msSanitizer to Detect Exceptions

msSanitizer is an exception detection tool based on AI processor. It provides memory check, contention check, uninitialization check, and synchronization check in single-operator development scenarios.

- Memory check: During operator development, the tool can locate memory problems such as illegal read/write, multi-core corruption, non-aligned access, memory leak, and illegal release. In addition, the tool can detect the memory of the CANN software stack, helping users locate the module with memory exception in the software stack.

- Contention check: The tool helps users locate data contention problems that may be caused by contention risks, including intra-core contention and inter-core contention. Intra-core contention includes inter-pipeline contention and intra-pipeline contention.

- Uninitialization check: The tool helps users locate dirty data read problems that may be caused by uninitialized memory. 

![](../raw/95ff0a306791d19e.png)

The msSanitizer tool does not support the detection of multi-thread operators and level-2 pointer operators.

For details, see [Operator Development Tools](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/optool/docs/en/quick_start/mskpp_quick_start.md).

**This function is supported only in the following scenarios:**

- Call operators through the method in [Kernel Launch Based on a Sample Project](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0056.html).

- Calling operators through [single-operator APIs](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0070.html).

- Calling a single-operator API (**aclnnxxx**) indirectly: single-operator calling in the PyTorch framework.

#### Using msDebug for Operator Debugging

msDebug is an operator debugging tool designed for Ascend devices. It is used to debug operator programs running on the NPU, providing debugging methods for operator developers. Currently, it supports only program debugging in the SIMD programming scenario. msDebug can debug all Ascend operators, including Ascend C operators (Vector, Cube, and fused operators). The specific functions include breakpoint setting, variable and memory printing, single-step debugging, running interruption, core switching, program status check, debugging information display, and core dump file parsing. You can select the functions as needed. For details, see [Operator Development Tools](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/optool/docs/en/quick_start/mskpp_quick_start.md).

- Call operators through the method in [Kernel Launch Based on a Sample Project](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0056.html).

- Calling operators through [single-operator APIs](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0070.html).

- Calling a single-operator API (**aclnnxxx**) indirectly: single-operator calling in the PyTorch framework.

 **Parent topic:** [Function Debugging](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00058.html)
