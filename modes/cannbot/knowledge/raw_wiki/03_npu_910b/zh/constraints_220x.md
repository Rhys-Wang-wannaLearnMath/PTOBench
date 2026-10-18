> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../en/constraints_220x.md); [official page](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00048.html); [unaltered source body](../raw/b966b7e4215b309d.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="ZH-CN_TOPIC_0000002531522238"></a><a name="ZH-CN_TOPIC_0000002531522238"></a> 

# NPU架构版本220x

本节介绍硬件约束以及解决方案建议。对应的产品型号为：

- Atlas A3 训练系列产品/Atlas A3 推理系列产品

- Atlas A2 训练系列产品/Atlas A2 推理系列产品

<table><b>表1 </b>硬件约束以及解决方案建议<thead><tr><th><p>分类</p>
</th>
<th><p>硬件约束描述</p>
</th>
<th><p>解决方案建议</p>
</th>
</tr>
</thead>
<tbody><tr><td><p>内存访问（L0 Buffer/L1 Buffer/UB等）</p>
</td>
<td><p>各存储单元的最小访问粒度/地址对齐要求：</p>
<p>Unified Buffer：32Byte对齐。</p>
<p>L1 Buffer：32Byte对齐。</p>
<p>L0A Buffer/L0B Buffer：512Byte对齐。</p>
<p>L0C Buffer：64Byte对齐。</p>
<p>BiasTable Buffer：64Byte对齐。</p>
<p>Fixpipe Buffer：128Byte对齐。</p>
</td>
<td><ul><li>进行数据搬运时，需要感知对齐约束。</li><li>针对UB，遇到一些非对齐的场景，可以使用非对齐搬运的接口或者通过一些技巧（比如搬入时包含冗余数据，搬出时去除冗余数据）来解决。详情见<a href="https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0034.html">非对齐场景</a>。</li></ul>
</td>
</tr>
<tr><td><p>内存访问（UB）</p>
</td>
<td><p>UB bank访问冲突（Vector计算访问/搬运访问）。</p>
<p></p>
</td>
<td><p>需要按照芯片要求，在软件实现时错开处理的地址，从而解决bank冲突。具体解决方案可参考<a href="https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_0025.html">避免Unified Buffer的bank冲突</a>章节。</p>
</td>
</tr>
<tr><td><p>内存访问（GM）</p>
</td>
<td><p>多核并行同地址访问GM，会被硬件串行化。</p>
<p></p>
</td>
<td><p>对相同地址的访问，会被硬件串行化，性能为排队时间，大约下降10%-20%；</p>
<p>多核访问通过错峰访问（调整数据访问顺序和修改切分策略等），使得第一次加载数据到L2 Cache，后续访问性能反而提升。具体解决方案可参考<a href="https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00013.html">避免同地址访问</a>章节。</p>
</td>
</tr>
<tr><td><p>内存访问（GM）</p>
</td>
<td><p>单次搬运数据长度16KB以上时，可发挥带宽的最佳性能。</p>
</td>
<td><p>根据实测经验，单次搬运数据长度16KB以上时，通常能较好地发挥出带宽的最佳性能。因此对于单次搬运，应考虑尽可能的搬运较大的数据块（不同芯片不一样）。</p>
<p>具体解决方案可参考<a href="https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_0013.html">尽量一次搬运较大的数据块</a>章节。</p>
</td>
</tr>
<tr><td><p>内存访问（GM--&gt;L1）</p>
</td>
<td><p>DataCopy中源操作数相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔）不要超出65535，单位为DataBlock（32字节）。</p>
</td>
<td><p>前面一个数据块的尾与后面数据块的头的间隔超出65535时，需要拆分成多条指令来实现。</p>
</td>
</tr>
<tr><td><p>内存访问（GM）</p>
</td>
<td><p>数据搬运会被拆成128B/256B/512B不同长度进行搬运，非对齐会向上取整。</p>
</td>
<td><p>Tiling尽量让搬运的内轴128B、256B、512B对齐。</p>
<p>具体解决方案可参考<a href="https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_0014.html">GM地址尽量512B对齐</a>章节。</p>
</td>
</tr>
<tr><td><p>Cube</p>
</td>
<td><p>MTE1和MMAD指令队列的深度为32。</p>
</td>
<td><p>对应指令队列容易满，会阻塞其他指令下发，引起流水断流。</p>
<p>Load2D从L1 Buffer搬运到L0 Buffer，需要发射32条指令，Load3D只需要一条指令，建议使用Load3D来实现。</p>
</td>
</tr>
<tr><td><p>ICache</p>
</td>
<td><p>ICache硬件规格限制32KB。</p>
</td>
<td><p>拆分Tiling_key或使用模板函数来减少代码段。详情见<a href="https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00025.html">Tiling模板编程</a>。</p>
</td>
</tr>
<tr><td><p>ICache</p>
</td>
<td><p>多核并行同地址访问ICache，会被硬件串行化。</p>
<p></p>
</td>
<td><p>小Shape场景尽量减少启动核数，减少多核同地址访问问题。</p>
</td>
</tr>
<tr><td><p>DCache</p>
</td>
<td><p>DCache硬件规格限制32KB</p>
</td>
<td><p>无</p>
</td>
</tr>
<tr><td><p>Scalar</p>
</td>
<td><p>标量写GM时，数据会被缓存在Dcache。硬件不保证DCache和GM一致性，需要用户保证。</p>
</td>
<td><p>使用DataCacheCleanAndInvalid来保证一致性。</p>
</td>
</tr>
<tr><td><p>Cube</p>
</td>
<td><p>L0C Buffer容量128KB。</p>
</td>
<td><p>无</p>
</td>
</tr>
<tr><td><p>Cube</p>
</td>
<td><p>BiasTable Buffer为1KB。</p>
</td>
<td><p>无</p>
</td>
</tr>
<tr><td><p>Cube</p>
</td>
<td><p>Cube计算场景float算力为half算力的1/4。</p>
</td>
<td><p>无</p>
</td>
</tr>
<tr><td><p>Cube</p>
</td>
<td><p>Cube输出随路量化场景，不支持int32_t到bfloat16_t类型量化。</p>
</td>
<td><ul><li>AIV上存在int32_t-&gt;float、float&gt;bfloat16_t的转换。</li><li>AIC支持float-&gt;bfloat16_t的随路量化。</li></ul>
</td>
</tr>
<tr><td><p>Vector</p>
</td>
<td><p>Reduce接口half比float性能差。</p>
</td>
<td><p>half写回UB时存在非32Byte对齐，导致性能劣化，建议把half转float计算，此场景建议使用float数据类型。</p>
</td>
</tr>
<tr><td><p>Vector</p>
</td>
<td><p>Exp/Ln接口处理同样数量的half/float的耗时是一样的</p>
</td>
<td><p>内部对float做了优化，故两者性能相当，开发者可以根据实际情况选择合适的精度类型。</p>
</td>
</tr>
<tr><td><p>流水同步（核内）</p>
</td>
<td><p>set/wait同步不匹配，状态会残留，影响后续算子。</p>
</td>
<td><p>通过孪生调试/mssanitizer工具，提前识别此类问题。</p>
</td>
</tr>
<tr><td><p>流水同步（核间）</p>
</td>
<td><p>CrossCoreSetFlag计数器存在限制，超出15次需要做一次反向同步，否则会出现卡死。</p>
</td>
<td><p>通过孪生调试/mssanitizer工具，超出场景提前报错。</p>
</td>
</tr>
<tr><td><p>API通用约束</p>
</td>
<td><p>使用Ascend C API时，源操作数和目的操作数的地址重叠通用约束</p>
</td>
<td><p>使用基础API的Tensor高维切分计算接口时，为了节省地址空间，开发者可以定义一个Tensor，供源操作数与目的操作数同时使用（即地址重叠）。使用时需要注意以下约束：</p>
<ul><li>单次迭代内：源操作数与目的操作数必须100%完全重叠，不支持部分重叠。</li><li>多次迭代间：不支持前序迭代的目的操作数与后序迭代的源操作数重叠。例如，第N次迭代的目的操作数是第N+1次的源操作数。在这种情况下，第N次迭代可能会改写覆盖源操作数的数值，导致无法得到预期结果。特别地，对于部分双目计算类的API（Add、Sub、Mul、Max、Min、AddRelu、SubRelu），当数据类型为half、int32_t、float时，支持前序迭代的目的操作数与后序迭代的源操作数重叠：仅针对目的操作数和第二个源操作数重叠的情况，且src1RepStride或者dstRepStride必须为0。</li></ul>
</td>
</tr>
</tbody>
</table>

**父主题：** [硬件约束](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00061.html)
