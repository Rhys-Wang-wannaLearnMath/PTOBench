> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../zh/tiling_headers.md); [official page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10047.html); [unaltered source body](../raw/18984ecf935e3464.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="EN-US_TOPIC_0000002531522186"></a><a name="EN-US_TOPIC_0000002531522186"></a> 

# How to Use the Header File on Which Tiling Depends

The Scalar Unit of the AI processor has limited execution capability. Therefore, to reduce scalar computations on the operator kernel, some computations are executed on the host. In this case, you need to compile the [tiling code on the host](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0064.html). Note that the dependent header file needs to be imported when the tiling API of the high-level APIs is called or the tiling structure parameter of the high-level APIs is used in the program. In different tiling implementation modes, the details are as follows:

- [Using the standard C++ syntax to define the tiling structure](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00024.html)

In this mode, the following dependent header file needs to be imported. The tiling structures of all high-level APIs are defined in the **AscendC::tiling** namespace. Therefore, you need to use **AscendC::tiling** to access the tiling structure of a specific API.

```cpp
#include "kernel_tiling/kernel_tiling.h"

// ...
AscendC::tiling::TCubeTiling cubeTilingData;
```

- [Using the TILING_DATA_DEF macro to define the tiling structure](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00023.html)

In this mode, the following dependent header file needs to be imported. The tiling structures and tiling functions of all high-level APIs are defined in the **optiling** namespace.

```cpp
#include "tiling/tiling_api.h"

namespace optiling {
// ...
}
```

**Parent topic:** [Quick Reference Guide to Common Operations](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10046.html)
