> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../en/tiling_headers.md); [official page](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10047.html); [unaltered source body](../raw/a040f854e95bb23f.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="ZH-CN_TOPIC_0000002531522186"></a><a name="ZH-CN_TOPIC_0000002531522186"></a> 

# 如何使用Tiling依赖的头文件

由于AI处理器的Scalar计算单元执行能力有限，为减少算子Kernel侧的Scalar计算，将部分计算在Host端执行，这需要编写[Host端Tiling代码](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0064.html)。注意，在程序中调用高阶API的Tiling接口或者使用高阶API的Tiling结构体参数时，需要引入依赖的头文件。在不同的Tiling实现方式下，具体为：

- [使用标准C++语法定义Tiling结构体](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00024.html)

这种方式需要引入依赖的头文件如下。所有高阶API的Tiling结构体定义在AscendC::tiling命名空间下，因此需要通过AscendC::tiling访问具体API的Tiling结构体。

```cpp
#include "kernel_tiling/kernel_tiling.h"

// ...
AscendC::tiling::TCubeTiling cubeTilingData;
```

- [使用TILING_DATA_DEF宏定义Tiling结构体](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00023.html)

这种方式需要引入依赖的头文件如下。所有高阶API的Tiling结构体和Tiling函数定义在optiling命名空间下。

```cpp
#include "tiling/tiling_api.h"

namespace optiling {
// ...
}
```

**父主题：** [常用操作速查指导](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10046.html)
