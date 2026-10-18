> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../en/cpu_twin_debugging.md); [official page](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0073.html); [unaltered source body](../raw/a7287b4f0bb0054f.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="ZH-CN_TOPIC_0000002562442139"></a><a name="ZH-CN_TOPIC_0000002562442139"></a> 

# CPU域孪生调试

本节介绍CPU域调试的方法：CPU侧验证核函数，gdb调试、使用printf命令打印。当前SIMT编程场景不支持。

![](../raw/1a9661ed95ef53fb.png)

CPU调测过程中，配置日志相关环境变量，可以记录程序的运行过程及异常信息，有助于开发者进行功能调测。

关于环境变量的使用约束以及详细说明，可参见[日志](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0118.html)。

#### CPU侧验证核函数

在非昇腾设备上，开发者可以利用CPU仿真环境先行进行算子开发和测试，并在准备就绪后，利用昇腾设备进行加速计算。在[编译与运行](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00042.html)章节，我们已经介绍了算子Kernel程序NPU域的编译运行。相比于NPU域的算子运行逻辑，CPU域调试将算子Kernel程序以Host程序的形式进行编译，此时算子Kernel程序链接CPU调测库，执行编译生成的可执行文件，可以完成算子CPU域的运行验证。CPU侧的运行程序，通过GDB通用调试工具进行单步调试，可以精准验证程序执行流程是否符合预期。

 **图1 **CPU域和NPU域的核函数运行逻辑对比 

![](../raw/54104d6b44475366.png)

推荐使用CMake编译方式，可在最小化修改的情况下快速开启CPU域孪生调试功能。

- 启用CPU域调试需包含"cpu_debug_launch.h"头文件。 

 bisheng编译器在CPU调试模式下会对<<<>>>调用核函数的过程进行转义，实现核函数在CPU域下的调用，相关调用函数定义在"cpu_debug_launch.h"中，在使用<<<>>>语法调用核函数的源文件中，请通过以下方式包含必需的头文件： 

```cpp
#ifdef ASCENDC_CPU_DEBUG
#include "cpu_debug_launch.h"
#endif
```

- 通过在CMake配置阶段传入变量CMAKE_ASC_RUN_MODE和CMAKE_ASC_ARCHITECTURES即可开启CPU域编译。命令示例如下： 

```cpp
cmake -B build -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201
```

cpu表示开启CPU域编译，dav-后为NPU架构版本号，请根据实际情况进行填写。

其他CMakeLists.txt项目配置[通过CMake编译](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00039.html)进行编写。

![](../raw/1a9661ed95ef53fb.png)

为了实现CPU域与NPU域代码归一，框架在CPU域中仅对部分acl接口进行适配，开发者在使用CPU域调测功能时，仅支持使用如下acl接口，并且不支持用户自行链接ascendcl库：

- 有实际功能接口，支持CPU域调用 

- aclDataTypeSize、aclFloat16ToFloat、aclFloatToFloat16。

- aclrtMalloc、aclrtFree、aclrtMallocHost、aclrtFreeHost、aclrtMemset、aclrtMemsetAsync、aclrtMemcpy、aclrtMemcpyAsync、aclrtMemcpy2d、aclrtMemcpy2dAsync、aclrtCreateContext、aclrtDestroyContext。

- 无实际功能接口，打桩实现。 

- Profiling数据采集 

<a name="ZH-CN_TOPIC_0000002562442139__li2852942164218"></a><a name="li2852942164218"></a>aclprofInit、aclprofSetConfig、aclprofStart、aclprofStop、aclprofFinalize。

- 系统配置 

<a name="ZH-CN_TOPIC_0000002562442139__li1385214264217"></a><a name="li1385214264217"></a>aclInit、aclFinalize、aclrtGetVersion。

- 运行时管理 

<a name="ZH-CN_TOPIC_0000002562442139__li178528425428"></a><a name="li178528425428"></a>aclrtSetDevice、aclrtResetDevice、aclrtCreateStream、aclrtCreateStreamWithConfig、aclrtDestroyStream、aclrtDestroyStreamForce、aclrtSynchronizeStream、aclrtCreateContext、aclrtDestroyContext。

#### gdb调试

可使用**gdb**单步调试算子计算精度。由于cpu调测已转为多进程调试，每个核都会拉起独立的子进程，故gdb需要转换成子进程调试的方式。针对耦合架构，每个AI Core会拉起1个子进程。针对分离架构，默认每个AI Core会拉起3个子进程，1个Cube，2个Vector。

- 调试单独一个子进程 

<a name="ZH-CN_TOPIC_0000002562442139__zh-cn_topic_0000001457309966_li1379314003611"></a><a name="zh-cn_topic_0000001457309966_li1379314003611"></a>启动gdb，示例中的add_custom_cpu为CPU域的算子可执行文件，参考[修改并执行一键式编译运行脚本](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0056.html#ZH-CN_TOPIC_0000002531522228__section188001652105215)，将一键式编译运行脚本中的run-mode设置成cpu，即可编译生成CPU域的算子可执行文件。

 gdb启动后，首先设置跟踪子进程，之后再打断点，就会停留在子进程中，但是这种方式只会停留在遇到断点的第一个子进程中，其余子进程和主进程会继续执行直到退出。涉及到核间同步的算子无法使用这种方法进行调试。 

```cpp
gdb --args add_custom_cpu  // 启动gdb，add_custom_cpu为算子可执行文件
(gdb) set follow-fork-mode child
```

- 调试多个子进程 

<a name="ZH-CN_TOPIC_0000002562442139__zh-cn_topic_0000001457309966_li17465575361"></a><a name="zh-cn_topic_0000001457309966_li17465575361"></a>如果涉及到核间同步，那么需要能同时调试多个子进程。

在gdb启动后，首先设置调试模式为只调试一个进程，挂起其他进程。设置的命令如下：

```cpp
(gdb) set detach-on-fork off
```

查看当前调试模式的命令为：

```cpp
(gdb) show detach-on-fork
```

中断gdb程序要使用捕捉事件的方式，即gdb程序捕捉fork这一事件并中断。这样在每一次起子进程时就可以中断gdb程序。设置的命令为：

```cpp
(gdb) catch fork
```

当执行r后，可以查看当前的进程信息：

```cpp
(gdb) info inferiors
  Num  Description
* 1    process 19613
```

可以看到，当第一次执行fork的时候，程序断在了主进程fork的位置，子进程还未生成。

执行c后，再次查看info inferiors，可以看到此时第一个子进程已经启动。

```cpp
(gdb) info inferiors
  Num  Description 
* 1    process 19613
  2    process 19626
```

这个时候可以使用切换到第二个进程，也就是第一个子进程，再打上断点进行调试，此时主进程是暂停状态：

```cpp
(gdb) inferior 2
[Switching to inferior 2 [process 19626] ($HOME/demo)]
(gdb) info inferiors
  Num  Description
  1    process 19613
* 2    process 19626
```

请注意，inferior后跟的数字是进程的序号，而不是进程号。

如果遇到同步阻塞，可以切换回主进程继续生成子进程，然后再切换到新的子进程进行调试，等到同步条件完成后，再切回第一个子进程继续执行。

如下是调试一个单独子进程的命令样例：

```cpp
gdb --args add_custom_cpu
set follow-fork-mode child
break add_custom.cpp:45
run
list
backtrace
print i
break add_custom.cpp:56
continue
display xLocal
quit
```

#### 使用printf打印命令打印

 在代码中直接编写printf(...)来观察数值的输出。样例代码如下： 

```cpp
printf("xLocal size: %d\n", xLocal.GetSize()); 
printf("tileLength: %d\n", tileLength);
```

 **父主题：** [功能调试](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00058.html)
