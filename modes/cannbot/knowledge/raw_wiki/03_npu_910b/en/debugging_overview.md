> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../zh/debugging_overview.md); [official page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0072.html); [unaltered source body](../raw/e6de3fa2598ea596.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="EN-US_TOPIC_0000002531362252"></a><a name="EN-US_TOPIC_0000002531362252"></a> 

# Overview

To debug an Ascend C operator, you need to first call the Ascend C class library to compile the kernel source code of the Ascend C operator, compile the kernel source code by using the GCC to generate common binary files on the CPU, and use the GDB debugging tool for debugging. In addition, you need to compile the kernel source code by using the BiSheng compiler to generate binary files on the NPU. You can use APIs such as **printf**/**assert** to print data, and use the simulation dotting or Profiling tool to collect board data for debugging.

![](../raw/2638fde7ff7eb113.png)

The following table lists the debugging and tuning methods and tools.

<table><b>Table 1 </b>Debugging and tuning methods and tools<thead><tr><th><p>Category</p>
</th>
<th><p>Subcategory</p>
</th>
<th><p>Method</p>
</th>
</tr>
</thead>
<tbody><tr><td rowspan="5"><p>Function debugging</p>
</td>
<td><p>CPU twin debugging</p>
</td>
<td><p><strong>Twin debugging</strong>: The same operator code can be used to debug accuracy on the CPU and performance on the NPU. In the CPU domain, you can perform GDB debugging and use the <strong>printf</strong> command to print information.</p>
</td>
</tr>
<tr><td rowspan="4"><p>Board debugging on the NPU</p>
</td>
<td><p><strong>printf</strong>/<strong>assert</strong>: <strong>printf</strong> is used to print scalar and string information. <strong>assert</strong> is used to set checkpoints in the code. When a condition is not met, the program terminates immediately and reports an error.</p>
</td>
</tr>
<tr><td><p><strong>DumpTensor</strong>: Use the <strong>DumpTensor</strong> API to print the data of a specified tensor. It is supported only in SIMD programming scenarios.</p>
</td>
</tr>
<tr><td><p><strong>Board debugging tool</strong>: Use the msDebug tool to debug the operator program running on the NPU. In a real hardware environment, test the input and output of the operator to verify whether the operator functions properly. The specific functions include setting breakpoints, printing variables and memory, performing single-step debugging, and interrupting execution. Currently, the SIMT programming scenario is not supported.</p>
</td>
</tr>
<tr><td><p><strong>Memory check tool</strong>: Use the msSanitizer tool to check the memory. It can detect and report memory access exceptions such as out-of-bounds and misalignment of the external storage (global memory) and internal storage (local memory) during operator running.</p>
</td>
</tr>
<tr><td><p>Performance tuning</p>
</td>
<td><p>-</p>
</td>
<td><p><strong>msProf tool</strong>:</p>
<p>The msProf performance analysis tool is used to collect and analyze key performance metrics of operators running on <span>AI processor</span>s. You can efficiently locate software and hardware performance bottlenecks of operators based on the output performance data, thereby enhancing the overall efficiency of operator performance analysis.</p>
<p>Performance data can currently be collected and automatically parsed based on various running modes (onboard or simulation) and file formats (executable files or operator binary <strong>.o</strong> files).</p>
</td>
</tr>
</tbody>
</table>

**Parent topic:** [Debugging and Tuning](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0071.html)
