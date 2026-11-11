> **CANN 9.0.X — local 910B4 / CANN 9.0.0 reading edition.** [Official English page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00020.html).

# Avoiding Bank Conflicts (NPU Architecture Version 220x)

[Priority] High

This performance optimization suggestion applies to the following product models:

- Atlas A3 training product / Atlas A3 inference product

- Atlas A2 training product / Atlas A2 inference product

[Description] To improve data access efficiency and throughput, the Unified Buffer adopts a bank (memory modules of the same size) structure. The Unified Buffer has a total size of 192 KB and is divided into 48 banks. Each bank consists of 128 rows, with each row being 32 bytes long. These 48 banks are further organized into 16 bank groups, with each bank group containing three banks. For example, bank15, bank31, and bank47 form a bank group.

Each bank can independently read and write data, and multiple data requests can be processed at the same time. However, when a plurality of read/write operations attempt to access a same bank or bank group at the same time, due to a limitation of hardware resources, these operations need to wait in a queue, resulting in a bank conflict and performance deterioration.

Specifically, the Vector Unit can read or write a row of data from or into each bank group in each cycle (an instruction cycle). If multiple operations in the same API attempt to access the same bank or bank group at the same time, the Vector Unit cannot process all requests in the same cycle. As a result, these requests wait in a queue. This queuing increases the data access latency and reduces the overall system performance.

> **Local configuration difference:** The installed 910B4 INI specifies `ubbank_num=64`, `ubbank_size=4096`, and `ubbank_group_num=16`, while the source below illustrates 48 banks in a 192 KiB UB. These fields alone do not establish the physical-to-visible bank mapping; `64 * 4096` is not an allocatable UB capacity. Retain the source examples as a bank-conflict model, and verify actual offsets with generated code and device profiling. See the [environment baseline](../../BASELINE.md).

#### Typical Scenarios of Bank Conflicts

Bank conflicts can be classified into the following three scenarios:

- **Read/Write conflict**: A read operation and a write operation attempt to access the same bank at the same time.

- **Write/Write conflict**: Multiple write operations attempt to access the same bank group at the same time.

- **Read/Read conflict**: Multiple read operations attempt to access the same bank group at the same time.

The following provides some specific examples. Assume that 0x10000 is on bank16, 0x10020 is on bank17, and 0x20020 is on bank33.

- Example of a read/write conflict

 When the source operand src and destination operand dst of the Vector instruction are read and written to the same bank at the same time, a read/write conflict occurs. The details are as follows:

| No. | src addr | dst addr | bank | bank group | Conclusion |
| --- | --- | --- | --- | --- | --- |
| Example 1 | 0x10020 | 0x10000 | bank_id0 != bank_id1 | bank_group_id0 != bank_group_id1 | The addresses of src and dst are in bank17 and bank16, respectively. Therefore, no conflict occurs. |
| Example 2 | 0x10020 | 0x10E20 | bank_id0 == bank_id1 | bank_group_id0 == bank_group_id1 | The addresses of src and dst are both in bank17. Therefore, a conflict occurs. |

- Example of a write/write conflict

 When eight data blocks (block0 to block7) corresponding to the destination operand dst of the Vector instruction are written to the same bank group, a write/write conflict occurs. The details are as follows:

| No. | dst addr | blk_stride | block0_addr | block1_addr | block2_addr | ... | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Example 1 | 0x1FE00 | 16 | 0x1FE00 | 0x20000 | 0x20200 | ... | All eight data blocks are in the same bank group. Therefore, conflicts occur. One repeat is written in eight cycles. |
| Example 2 | 0x1FE00 | 8 | 0x1FE00 | 0x1FF00 | 0x20000 | ... | block0 and block2 are in the same bank group and conflict with each other. One repeat is written in four cycles. |

- Read/Read conflict

- When multiple source operands of the Vector instruction are read to the same bank group at the same time, a read/read conflict occurs. The analysis is as follows:

| No. | src0 addr | src1 addr | bank | bank group | Conclusion |
| --- | --- | --- | --- | --- | --- |
| Example 1 | 0x10020 | 0x20020 | bank_id0 != bank_id1 | bank_group_id0 == bank_group_id1 | Conflict |
| Example 2 | 0x10020 | 0x10000 | bank_id0 != bank_id1 | bank_group_id0 != bank_group_id1 | No conflict |

- When the eight data blocks (block0 to block7) corresponding to a source operand of the Vector instruction are read to the same bank group, a read/read conflict occurs. The analysis is as follows:

| No. | src addr | blk_stride | block0_addr | block1_addr | block2_addr | ... | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Example 1 | 0x1FE00 | 16 | 0x1FE00 | 0x20000 | 0x20200 | ... | All eight data blocks are in the same bank group. Therefore, conflicts occur. One repeat is read in eight cycles. |
| Example 2 | 0x1FE00 | 8 | 0x1FE00 | 0x1FF00 | 0x20000 | ... | block0 and block2 are in the same bank group. Therefore, a conflict occurs. One repeat is completed in four cycles. |

The msProf tool can be used to collect profile data related to the resource conflict ratio.

For details about how to use the tool and the description of the profile data file, see [Operator Development Tools](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/optool/docs/en/quick_start/mskpp_quick_start.md).

#### How to Avoid Bank Conflicts

There are two methods to avoid bank conflicts: **optimizing the computation logic** and **optimizing address allocation**.

- **Optimizing the computation logic**

Perform the (1, 0, 2) transpose operation on an input with shape (8, 16, 16). Then, the output shape is (16, 8, 16). You can avoid conflicts by changing the computation logic from "skipped read and continuous write" to "continuous read and skipped write". The implementation solutions are compared as follows:

##### Original Implementation

Read skipping, continuous write

The eight data blocks input in the same repeat are in the same bank group, causing a read/read conflict.
```cpp
uint64_t mask = 128;
UnaryRepeatParams params;
params.dstBlkStride  = 1;
params.srcBlkStride = 16;
for(uint32_t i=0; i<16; i++)   {
    AscendC::Adds(dstLocal[i * 128], srcLocal[i * 16], 0, mask, 1, params);
}
```

##### Optimized Implementation

Continuous read, skip write

The eight data blocks input in the same repeat are not in the same bank group, avoiding a read/read conflict.
```cpp
uint64_t mask = 128;
UnaryRepeatParams params;
params.dstBlkStride  = 8;
params.srcBlkStride = 1;
for(uint32_t i=0; i<8; i++)   {
    AscendC::Adds(dstLocal[i * 16], srcLocal[i * 256], 0, mask, 2, params);
}
```

- **Optimizing address allocation**

Implement the addition of 4096 consecutive float elements (z = x + y). By allocating more memory, ensure that x and y do not appear in the same bank group at the same time, and x/y and z do not appear in the same bank at the same time within a repeat. For details about the complete example, see [sample of avoiding bank conflicts](https://gitee.com/ascend/samples/tree/master/operator/ascendc/4_best_practices/4_bank_conflict).

The implementation solutions are compared as follows:

##### Original Implementation

No address optimization is performed. InitBuffer is directly used to allocate memory. The addresses of each tensor are as follows:

x: The start address is 0x0, and the tensor length is 4096 x sizeof(float) bytes.

y: The start address is 0x4000, and the tensor length is 4096 x sizeof(float) bytes.

z: The start address is 0x8000, and the tensor length is 4096 x sizeof(float) bytes.

In a repeat, x and y read the same bank group at the same time, and x/y and z read and write the same bank at the same time.
```cpp
pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float));
pipe.InitBuffer(inQueueY, 1, 4096 * sizeof(float));
pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));
```

##### Optimized Implementation

Optimize the address. When InitBuffer is used to allocate memory, properly allocate more memory. The addresses of each tensor are as follows:

x: The start address is 0x0, and the tensor length is (4096 x sizeof(float) + 256) bytes.

y: The start address is 0x4100, and the tensor length is (64 x 1024 – (4096 x sizeof(float) + 256)) bytes.

z: The start address is 0x10000, and the tensor length is 4096 x sizeof(float) bytes.

Allocate 256 bytes more for x to prevent x and y from reading the same bank group in a repeat. Allocate more space for y to ensure that z does not fall into the same bank as x or y.
```cpp
pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float) + 256); // Allocate 256 bytes more.
pipe.InitBuffer(inQueueY, 1, 64 * 1024 - (4096 * sizeof(float) + 256)); // Allocate more space to ensure that z does not fall into the same bank as x/y. 64 * 1024 is the space of 16 banks, and 4096 * sizeof(float) + 256 is the space occupied by x.
pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));
```
