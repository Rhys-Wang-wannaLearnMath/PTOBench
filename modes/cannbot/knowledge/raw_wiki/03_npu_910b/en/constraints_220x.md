> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../zh/constraints_220x.md); [official page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00048.html); [unaltered source body](../raw/51e599e1a46e9c41.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="EN-US_TOPIC_0000002531522238"></a><a name="EN-US_TOPIC_0000002531522238"></a> 

# NPU Architecture Version 220x

This section describes the hardware constraints and recommended solutions. The corresponding product models are as follows:

- Atlas A3 training product/Atlas A3 inference product

- Atlas A2 training product/Atlas A2 inference product

<table><b>Table 1 </b>Hardware constraints and recommended solutions<thead><tr><th><p>Category</p>
</th>
<th><p>Hardware Constraint</p>
</th>
<th><p>Recommended Solution</p>
</th>
</tr>
</thead>
<tbody><tr><td><p>Memory access (L0 Buffer/L1 Buffer/UB)</p>
</td>
<td><p>The minimum access granularities/alignment requirements of each storage unit are as follows:</p>
<p>Unified Buffer: 32-byte aligned</p>
<p>L1 Buffer: 32-byte aligned</p>
<p>L0A Buffer/L0B Buffer: 512-byte aligned</p>
<p>L0C Buffer: 64-byte aligned</p>
<p>BiasTable Buffer: 64-byte aligned</p>
<p>FixPipe Buffer: 128-byte aligned</p>
</td>
<td><ul><li>During data movement, alignment constraints need to be considered.</li><li>For the UB, in some non-aligned scenarios, you can use the non-aligned movement API or use some techniques (for example, including redundant data during move-in and removing redundant data during move-out) to solve the problem. For details, see <a href="https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0034.html">Non-Alignment Scenario</a>.</li></ul>
</td>
</tr>
<tr><td><p>Memory access (UB)</p>
</td>
<td><p>UB bank access conflicts (vector computation access/movement access).</p>
<p></p>
</td>
<td><p>The addresses need to be staggered during software implementation based on the chip requirements to resolve bank conflicts. For details, see <a href="https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_0025.html">Avoiding Bank Conflicts in the Unified Buffer</a>.</p>
</td>
</tr>
<tr><td><p>Memory access (GM)</p>
</td>
<td><p>When multiple cores concurrently access the GM at the same address, the access is serialized by hardware.</p>
<p></p>
</td>
<td><p>Access to the same address is serialized by hardware, and the performance (queuing time) decreases by about 10% to 20%.</p>
<p>Multi-core access is staggered (by adjusting the data access sequence and modifying the tiling policy) so that the first data load to the L2 cache improves subsequent access performance. For details, see <a href="https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00013.html">Avoiding Same-Address Access</a>.</p>
</td>
</tr>
<tr><td><p>Memory access (GM)</p>
</td>
<td><p>When the length of data moved at a time is greater than 16 KB, the optimal bandwidth performance can be achieved.</p>
</td>
<td><p>According to the test experience, optimal bandwidth performance can be achieved when data to be moved at a time is greater than 16 KB. Therefore, it is recommended that a large data block be moved at a time. (The size varies with the chip.)</p>
<p>For details, see <a href="https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_0013.html">Transferring a Large Data Block at a Time</a>.</p>
</td>
</tr>
<tr><td><p>Memory access (GM--&gt;L1)</p>
</td>
<td><p>The interval between adjacent data blocks of the source operand in DataCopy (the interval between the tail of the previous data block and the head of the next data block) does not exceed 65535. The unit is a data block (32 bytes).</p>
</td>
<td><p>If the interval between the tail of the previous data block and the head of the next data block exceeds 65535, the instruction needs to be split into multiple ones.</p>
</td>
</tr>
<tr><td><p>Memory access (GM)</p>
</td>
<td><p>Data is 128-byte, 256-byte, and 512-byte aligned for movement. If the data is not aligned, the length is rounded up.</p>
</td>
<td><p>Ensure that the inner axis of the tiling is 128-byte, 256-byte, or 512-byte aligned.</p>
<p>For details, see <a href="https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_0014.html">Using 512-Byte Alignment for the GM Address</a>.</p>
</td>
</tr>
<tr><td><p>Cube</p>
</td>
<td><p>The depth of the MTE1 and MMAD instruction queues is 32.</p>
</td>
<td><p>The corresponding instruction queue is prone to be full, which blocks the delivery of other instructions and causes pipeline interruption.</p>
<p>To move data from the L1 Buffer to the L0 Buffer using Load2D, 32 instructions need to be transmitted. However, Load3D requires only one instruction to implement data movement. Therefore, Load3D is recommended.</p>
</td>
</tr>
<tr><td><p>iCache</p>
</td>
<td><p>The iCache hardware specification is limited to 32 KB.</p>
</td>
<td><p>Split the Tiling_key or use template functions to reduce code segments. For details, see <a href="https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00025.html">Tiling Template Programming</a>.</p>
</td>
</tr>
<tr><td><p>iCache</p>
</td>
<td><p>When multiple cores concurrently access the iCache at the same address, the access is serialized by hardware.</p>
<p></p>
</td>
<td><p>In small-shape scenarios, reduce the number of started cores to the greatest extent to minimize the issue of concurrent access to the same address by multiple cores.</p>
</td>
</tr>
<tr><td><p>DCache</p>
</td>
<td><p>The DCache hardware specification is limited to 32 KB.</p>
</td>
<td><p>None</p>
</td>
</tr>
<tr><td><p>Scalar</p>
</td>
<td><p>When scalar data is written to the GM, the data is cached in the DCache. The hardware does not ensure the consistency between the DCache and GM. You need to ensure the consistency.</p>
</td>
<td><p>Use <strong>DataCacheCleanAndInvalid</strong> to ensure consistency.</p>
</td>
</tr>
<tr><td><p>Cube</p>
</td>
<td><p>The L0C Buffer capacity is 128 KB.</p>
</td>
<td><p>None</p>
</td>
</tr>
<tr><td><p>Cube</p>
</td>
<td><p>The BiasTable Buffer is 1 KB.</p>
</td>
<td><p>None</p>
</td>
</tr>
<tr><td><p>Cube</p>
</td>
<td><p>In cube computation scenarios, the float computing power is 1/4 of the half computing power.</p>
</td>
<td><p>None</p>
</td>
</tr>
<tr><td><p>Cube</p>
</td>
<td><p>In the cube output channel quantization scenario, quantization from int32_t to bfloat16_t is not supported.</p>
</td>
<td><ul><li>The AIV involves conversions from int32_t to float and from float to bfloat16_t.</li><li>The AIC supports channel quantization from float to bfloat16_t.</li></ul>
</td>
</tr>
<tr><td><p>Vector</p>
</td>
<td><p>For the Reduce API, the performance of half data is poorer than that of float data.</p>
</td>
<td><p>When half data is written back to the UB, it is not 32-byte aligned, resulting in performance deterioration. It is recommended that half data be converted to float data for computation. In this scenario, the float data type is recommended.</p>
</td>
</tr>
<tr><td><p>Vector</p>
</td>
<td><p>The time taken by the Exp/Ln API to process the same amount of half/float data is the same.</p>
</td>
<td><p>The float data type has been optimized internally, so the performance of the two data types is similar. You can select a proper precision type as needed.</p>
</td>
</tr>
<tr><td><p>Pipeline synchronization (intra-core)</p>
</td>
<td><p>If the set/wait synchronization does not match, the status remains, affecting subsequent operators.</p>
</td>
<td><p>Use the twin debugging/mssanitizer tool to identify such issues in advance.</p>
</td>
</tr>
<tr><td><p>Pipeline synchronization (inter-core)</p>
</td>
<td><p>The <strong>CrossCoreSetFlag</strong> counter has a limit. If the number of times exceeds 15, reverse synchronization is required. Otherwise, the system may be suspended.</p>
</td>
<td><p>Use the twin debugging/mssanitizer tool to report errors in advance in scenarios where the limit is exceeded.</p>
</td>
</tr>
<tr><td><p>General API restrictions</p>
</td>
<td><p>General restrictions on the overlapping of the source and destination operand addresses when using Ascend C APIs</p>
</td>
<td><p>To save memory space when using high-dimensional tensor sharding compute APIs of basic APIs, you can define a tensor shared by the source and destination operands (by address overlapping). Pay attention to the following restrictions when using this:</p>
<ul><li>In a single repeat, the source operand must completely overlap the destination operand. Partial overlapping is not supported.</li><li>Among multiple repeats, the destination operand of a previous repeat cannot overlap the source operand of a subsequent repeat. For example, the destination operand of the Nth repeat is the source operand of the (N+1)th repeat. In this case, the N th repeat may overwrite the value of the source operand, resulting in an unexpected result. In particular, for some two-operand compute APIs (<strong>Add</strong>, <strong>Sub</strong>, <strong>Mul</strong>, <strong>Max</strong>, <strong>Min</strong>, <strong>AddRelu</strong>, and <strong>SubRelu</strong>), when the data type is half, int32_t, or float, the destination operand of a previous repeat can overlap the source operand of a subsequent repeat. This is only applicable when the destination operand overlaps the second source operand, and <strong>src1RepStride</strong> or <strong>dstRepStride</strong> must be <strong>0</strong>.</li></ul>
</td>
</tr>
</tbody>
</table>

**Parent topic:** [Hardware Constraints](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00061.html)
