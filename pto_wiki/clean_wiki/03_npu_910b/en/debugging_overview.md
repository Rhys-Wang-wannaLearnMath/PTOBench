> **CANN 9.0.X — local 910B4 / CANN 9.0.0 reading edition.** [Official English page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0072.html).

# Overview

> **Local debugging scope:** This is the Ascend C CPU twin flow, which differs from simpler/PyPTO functional simulation. CANN CPU twin headers and the `dav-2201` CMake mode are installed, but `gdb` is not found on the current PATH. The GDB examples therefore require that prerequisite; simulator correctness does not certify device ordering or instruction profiling.

To debug an Ascend C operator, you need to first call the Ascend C class library to compile the kernel source code of the Ascend C operator, compile the kernel source code by using the GCC to generate common binary files on the CPU, and use the GDB debugging tool for debugging. In addition, you need to compile the kernel source code by using the BiSheng compiler to generate binary files on the NPU. You can use APIs such as **printf**/**assert** to print data, and use the simulation dotting or Profiling tool to collect board data for debugging.

The following table lists the debugging and tuning methods and tools.

| Category | Subcategory | Method |
| --- | --- | --- |
| Function debugging | CPU twin debugging | **Twin debugging**: The same operator code can be used to debug accuracy on the CPU and performance on the NPU. In the CPU domain, you can perform GDB debugging and use the **printf** command to print information. |
| Function debugging | Board debugging on the NPU | **printf**/**assert**: **printf** is used to print scalar and string information. **assert** is used to set checkpoints in the code. When a condition is not met, the program terminates immediately and reports an error. |
| Function debugging | Board debugging on the NPU | **DumpTensor**: Use the **DumpTensor** API to print the data of a specified tensor. It is supported only in SIMD programming scenarios. |
| Function debugging | Board debugging on the NPU | **Board debugging tool**: Use the msDebug tool to debug the operator program running on the NPU. In a real hardware environment, test the input and output of the operator to verify whether the operator functions properly. The specific functions include setting breakpoints, printing variables and memory, performing single-step debugging, and interrupting execution. Currently, the SIMT programming scenario is not supported. |
| Function debugging | Board debugging on the NPU | **Memory check tool**: Use the msSanitizer tool to check the memory. It can detect and report memory access exceptions such as out-of-bounds and misalignment of the external storage (global memory) and internal storage (local memory) during operator running. |
| Performance tuning | - | **msProf tool**: The msProf performance analysis tool is used to collect and analyze key performance metrics of operators running on AI processors. You can efficiently locate software and hardware performance bottlenecks of operators based on the output performance data, thereby enhancing the overall efficiency of operator performance analysis. Performance data can currently be collected and automatically parsed based on various running modes (onboard or simulation) and file formats (executable files or operator binary **.o** files). |
