> **CANN 9.0.X — local 910B4 / CANN 9.0.0 reading edition.** Official English page (`https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10047.html`).

# How to Use the Header File on Which Tiling Depends

> **Installed include layouts:** Both original includes below are present through the compatibility include root `$ASCEND_HOME_PATH/aarch64-linux/ascendc/include/highlevel_api`. With the newer `$ASCEND_HOME_PATH/aarch64-linux/asc/include` root, use `adv_api/kernel_tiling.h` for the first example; `tiling/tiling_api.h` remains a symlink to `adv_api/tiling_api.h`. Keep the original examples when using the compatibility root. Other transitive headers and libraries still depend on the CANN project build setup.

The Scalar Unit of the AI processor has limited execution capability. Therefore, to reduce scalar computations on the operator kernel, some computations are executed on the host. In this case, you need to compile the tiling code on the host (`https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0064.html`). Note that the dependent header file needs to be imported when the tiling API of the high-level APIs is called or the tiling structure parameter of the high-level APIs is used in the program. In different tiling implementation modes, the details are as follows:

- Using the standard C++ syntax to define the tiling structure (`https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00024.html`)

In this mode, the following dependent header file needs to be imported. The tiling structures of all high-level APIs are defined in the **AscendC::tiling** namespace. Therefore, you need to use **AscendC::tiling** to access the tiling structure of a specific API.
```cpp
#include "kernel_tiling/kernel_tiling.h"

// ...
AscendC::tiling::TCubeTiling cubeTilingData;
```

- Using the TILING_DATA_DEF macro to define the tiling structure (`https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00023.html`)

In this mode, the following dependent header file needs to be imported. The tiling structures and tiling functions of all high-level APIs are defined in the **optiling** namespace.
```cpp
#include "tiling/tiling_api.h"

namespace optiling {
// ...
}
```
