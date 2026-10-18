> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../en/bank_conflicts_220x.md); [official page](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00020.html); [unaltered source body](../raw/280be4665a5b3228.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="ZH-CN_TOPIC_0000002531509894"></a><a name="ZH-CN_TOPIC_0000002531509894"></a> 

# 避免bank冲突（NPU架构版本220x）

【优先级】高

![](../raw/1a9661ed95ef53fb.png)

该性能优化建议适用于如下产品型号：

- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

- Atlas A2 训练系列产品 / Atlas A2 推理系列产品

【描述】为了提高数据访问的效率和吞吐量，Unified Buffer采用了bank（大小相等的内存模块）结构设计。Unified Buffer总大小为192K，划分为48个bank。每个bank由128行组成，每行长度为32B。这48个bank进一步组织为16个bank group，每个bank group包含3个bank，例如bank15、bank31和bank47组成一个bank group。

 **图1 **bank结构示意图（图中箭头方向表示内存排布的顺序） 

![](../raw/38c5ca133daddc1f.png)

每个bank可以独立地进行数据的读写操作，允许多个数据请求同时进行。然而，当多个读写操作试图同时访问同一个bank或bank group时，由于硬件资源的限制，这些操作必须排队等待，会导致bank冲突，引起性能下降。

具体来说，Vector计算单元每拍（一个指令周期）能够从每个bank group中读取或写入一行数据。如果同一个API中的多个操作试图同时访问同一个bank或bank group，Vector计算单元无法在同一个周期内处理所有请求，导致这些请求排队等待。这种排队增加了数据访问的延迟，降低了系统的整体性能。

#### bank冲突的典型场景

bank冲突主要可以分为以下三种场景：

- **读写冲突**：读操作和写操作同时尝试访问同一个bank。

- **写写冲突**：多个写操作同时尝试访问同一个bank group。

- **读读冲突**：多个读操作同时尝试访问同一个bank group。

下文给出了一些具体的示例，假设，0x10000地址在bank16上，0x10020在bank17上，0x20020在bank33上，如下图所示：

 **图2 **地址分配示意图 

![](../raw/b335766696960a89.png)

- 读写冲突示例 

 <a name="ZH-CN_TOPIC_0000002531509894__li1987054183914"></a><a name="li1987054183914"></a>Vector指令的源操作数src和目的操作数dst同时读写到同一个bank时造成读写冲突，具体分析如下： 

<table>

          <b>表1 </b>读写冲突示例

         <thead>
          <tr>
           <th>
            <p>序号</p></th>
           <th>
            <p>src地址</p></th>
           <th>
            <p>dst地址</p></th>
           <th>
            <p>bank</p></th>
           <th>
            <p>bank group</p></th>
           <th>
            <p>结论</p></th>
          </tr>
         </thead>
         <tbody>
          <tr>
           <td>
            <p>示例1</p></td>
           <td>
            <p>0x10020</p></td>
           <td>
            <p>0x10000</p></td>
           <td>
            <p>bank_id0 != bank_id1</p></td>
           <td>
            <p>bank_group_id0 != bank_group_id1</p></td>
           <td>
            <p>src地址和dst地址分别属于bank16和bank17，故无冲突。</p></td>
          </tr>
          <tr>
           <td>
            <p>示例2</p></td>
           <td>
            <p>0x10020</p></td>
           <td>
            <p>0x10E20</p></td>
           <td>
            <p>bank_id0 == bank_id1</p></td>
           <td>
            <p>bank_group_id0 == bank_group_id1</p></td>
           <td>
            <p>src地址和dst地址的地址都在bank17，故存在冲突。</p></td>
          </tr>
         </tbody>
        </table>

- 写写冲突示例 

 <a name="ZH-CN_TOPIC_0000002531509894__li1264151618456"></a><a name="li1264151618456"></a>Vector指令目的操作数dst对应的8个DataBlock（block0-block7）同时写到一个bank group时造成写写冲突，具体分析如下： 

<table>

          <b>表2 </b>写写冲突示例

         <thead>
          <tr>
           <th>
            <p>序号</p></th>
           <th>
            <p>dst地址</p></th>
           <th>
            <p>blk_stride</p></th>
           <th>
            <p>block0_addr</p></th>
           <th>
            <p>block1_addr</p></th>
           <th>
            <p>block2_addr</p></th>
           <th>
            <p>...</p></th>
           <th>
            <p>结论</p></th>
          </tr>
         </thead>
         <tbody>
          <tr>
           <td>
            <p>示例1</p></td>
           <td>
            <p>0x1FE00</p></td>
           <td>
            <p>16</p></td>
           <td>
            <p>0x1FE00</p></td>
           <td>
            <p>0x20000</p></td>
           <td>
            <p>0x20200</p></td>
           <td>
            <p>...</p></td>
           <td>
            <p>8个DataBlock均在一个bank group下，故全部冲突，8拍完成一个Repeat的写入。</p></td>
          </tr>
          <tr>
           <td>
            <p>示例2</p></td>
           <td>
            <p>0x1FE00</p></td>
           <td>
            <p>8</p></td>
           <td>
            <p>0x1FE00</p></td>
           <td>
            <p>0x1FF00</p></td>
           <td>
            <p>0x20000</p></td>
           <td>
            <p>...</p></td>
           <td>
            <p>block0和block2在一个bank group，存在冲突，4拍完成一个Repeat的写入。</p></td>
          </tr>
         </tbody>
        </table>

- 读读冲突 

- Vector指令多个源操作数同时读到同一个bank group时造成读读冲突，具体分析如下： 

<table>

           <b>表3 </b>双src场景读读冲突示例

          <thead>
           <tr>
            <th>
             <p>序号</p></th>
            <th>
             <p>src0地址</p></th>
            <th>
             <p>src1地址</p></th>
            <th>
             <p>bank</p></th>
            <th>
             <p>bank group</p></th>
            <th>
             <p>结论</p></th>
           </tr>
          </thead>
          <tbody>
           <tr>
            <td>
             <p>示例1</p></td>
            <td>
             <p>0x10020</p></td>
            <td>
             <p>0x20020</p></td>
            <td>
             <p>bank_id0 != bank_id1</p></td>
            <td>
             <p>bank_group_id0 == bank_group_id1</p></td>
            <td>
             <p>存在冲突。</p></td>
           </tr>
           <tr>
            <td>
             <p>示例2</p></td>
            <td>
             <p>0x10020</p></td>
            <td>
             <p>0x10000</p></td>
            <td>
             <p>bank_id0 != bank_id1</p></td>
            <td>
             <p>bank_group_id0 != bank_group_id1</p></td>
            <td>
             <p>无冲突。</p></td>
           </tr>
          </tbody>
         </table>

- Vector指令某一个源操作数对应的8个DataBlock（block0-block7）读到同一个bank group时造成读读冲突，具体分析如下： 

<table>

           <b>表4 </b>单src场景读读冲突示例

          <thead>
           <tr>
            <th>
             <p>序号</p></th>
            <th>
             <p>src地址</p></th>
            <th>
             <p>blk_stride</p></th>
            <th>
             <p>block0_addr</p></th>
            <th>
             <p>block1_addr</p></th>
            <th>
             <p>block2_addr</p></th>
            <th>
             <p>...</p></th>
            <th>
             <p>结论</p></th>
           </tr>
          </thead>
          <tbody>
           <tr>
            <td>
             <p>示例1</p></td>
            <td>
             <p>0x1FE00</p></td>
            <td>
             <p>16</p></td>
            <td>
             <p>0x1FE00</p></td>
            <td>
             <p>0x20000</p></td>
            <td>
             <p>0x20200</p></td>
            <td>
             <p>...</p></td>
            <td>
             <p>8个DataBlock均在一个bank group下，故全部冲突，8拍完成一个Repeat的读操作。</p></td>
           </tr>
           <tr>
            <td>
             <p>示例2</p></td>
            <td>
             <p>0x1FE00</p></td>
            <td>
             <p>8</p></td>
            <td>
             <p>0x1FE00</p></td>
            <td>
             <p>0x1FF00</p></td>
            <td>
             <p>0x20000</p></td>
            <td>
             <p>...</p></td>
            <td>
             <p>block0和block2在同一个bank group下，存在冲突，4拍完成一个Repeat。</p></td>
           </tr>
          </tbody>
         </table>

![](../raw/1a9661ed95ef53fb.png)

通过msProf工具可以进行资源冲突占比的相关性能数据采集。

工具的具体使用方法和资源冲突占比文件性能数据文件说明请参考《[算子开发工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/devaids/optool/docs/zh/quick_start/mskpp_quick_start.md)》。

#### 如何避免bank冲突

避免bank冲突的方法有两种：**优化计算逻辑**和**优化地址分配**。

- **优化计算逻辑** 

对一个shape为(8, 16, 16)的输入做(1, 0, 2)的transpose操作，输出shape为(16, 8, 16)。通过将计算逻辑由“跳读，连续写”修改为“连续读，跳写”可避免冲突问题。实现方案对比如下：

<table>
        <thead>
         <tr>
          <th>
           <p>实现方案</p></th>
          <th>
           <p>原始实现</p></th>
          <th>
           <p>优化实现</p></th>
         </tr>
        </thead>
        <tbody>
         <tr>
          <td>
           <p>实现方法</p></td>
          <td>
           <p>跳读，连续写</p>
           <p>同一Repeat内输入的8个DataBlock都在同一个bank group而发生读读冲突。</p></td>
          <td>
           <p>连续读，跳写</p>
           <p>同一个Repeat内输入的8个DataBlock不在同一个bank group内，避免了读读冲突。</p></td>
         </tr>
         <tr>
          <td>
           <p>示意图</p></td>
          <td>
           <p><span></span></p></td>
          <td>
           <p><span></span></p></td>
         </tr>
         <tr>
          <td>
           <p>示例代码</p></td>
          <td>

             <table>
              <tbody>
               <tr>
                <td>

                  <pre><span>1</span>
<span>2</span>
<span>3</span>
<span>4</span>
<span>5</span>
<span>6</span>
<span>7</span></pre>
                 </td>
                <td>

                  <pre><span></span><span>uint64_t</span><span> </span><span>mask</span><span> </span><span>=</span><span> </span><span>128</span><span>;</span>
<span>UnaryRepeatParams</span><span> </span><span>params</span><span>;</span>
<span>params</span><span>.</span><span>dstBlkStride</span><span>  </span><span>=</span><span> </span><span>1</span><span>;</span>
<span>params</span><span>.</span><span>srcBlkStride</span><span> </span><span>=</span><span> </span><span>16</span><span>;</span>
<span>for</span><span>(</span><span>uint32_t</span><span> </span><span>i</span><span>=</span><span>0</span><span>;</span><span> </span><span>i</span><span>&lt;</span><span>16</span><span>;</span><span> </span><span>i</span><span>++</span><span>)</span><span>   </span><span>{</span>
<span>    </span><span>AscendC</span><span>::</span><span>Adds</span><span>(</span><span>dstLocal</span><span>[</span><span>i</span><span> </span><span>*</span><span> </span><span>128</span><span>],</span><span> </span><span>srcLocal</span><span>[</span><span>i</span><span> </span><span>*</span><span> </span><span>16</span><span>],</span><span> </span><span>0</span><span>,</span><span> </span><span>mask</span><span>,</span><span> </span><span>1</span><span>,</span><span> </span><span>params</span><span>);</span>
<span>}</span>
</pre>
                 </td>
               </tr>
              </tbody>
             </table>

           </td>
          <td>

             <table>
              <tbody>
               <tr>
                <td>

                  <pre><span>1</span>
<span>2</span>
<span>3</span>
<span>4</span>
<span>5</span>
<span>6</span>
<span>7</span></pre>
                 </td>
                <td>

                  <pre><span></span><span>uint64_t</span><span> </span><span>mask</span><span> </span><span>=</span><span> </span><span>128</span><span>;</span>
<span>UnaryRepeatParams</span><span> </span><span>params</span><span>;</span>
<span>params</span><span>.</span><span>dstBlkStride</span><span>  </span><span>=</span><span> </span><span>8</span><span>;</span>
<span>params</span><span>.</span><span>srcBlkStride</span><span> </span><span>=</span><span> </span><span>1</span><span>;</span>
<span>for</span><span>(</span><span>uint32_t</span><span> </span><span>i</span><span>=</span><span>0</span><span>;</span><span> </span><span>i</span><span>&lt;</span><span>8</span><span>;</span><span> </span><span>i</span><span>++</span><span>)</span><span>   </span><span>{</span>
<span>    </span><span>AscendC</span><span>::</span><span>Adds</span><span>(</span><span>dstLocal</span><span>[</span><span>i</span><span> </span><span>*</span><span> </span><span>16</span><span>],</span><span> </span><span>srcLocal</span><span>[</span><span>i</span><span> </span><span>*</span><span> </span><span>256</span><span>],</span><span> </span><span>0</span><span>,</span><span> </span><span>mask</span><span>,</span><span> </span><span>2</span><span>,</span><span> </span><span>params</span><span>);</span>
<span>}</span>
</pre>
                 </td>
               </tr>
              </tbody>
             </table>

           </td>
         </tr>
        </tbody>
       </table>

- **优化地址分配** 

实现连续4096个float元素的加法z = x + y，通过在内存分配时适当扩大内存，保证在一个Repeat内，x和y不会同时出现在同一个bank group内，x/y和z不会同时出现同一个bank内。完整样例可参考[避免bank冲突样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/4_best_practices/4_bank_conflict)。

实现方案对比如下：

<table>
        <thead>
         <tr>
          <th>
           <p>实现方案</p></th>
          <th>
           <p>原始实现</p></th>
          <th>
           <p>优化实现</p></th>
         </tr>
        </thead>
        <tbody>
         <tr>
          <td>
           <p>实现方法</p></td>
          <td>
           <p>不做地址优化，直接使用InitBuffer分配内存，各个Tensor的地址分别为：</p>
           <p>x：起始地址0x0，tensor长度为4096 * sizeof(float)字节</p>
           <p>y：起始地址0x4000，tensor长度为4096 * sizeof(float)字节</p>
           <p>z：起始地址0x8000，tensor长度为4096 * sizeof(float)字节</p>
           <p>在一个Repeat内，x和y同时读同一个bank group，x/y和z同时读写同一个bank。</p></td>
          <td>
           <p>优化地址，使用InitBuffer分配内存时适当扩大内存申请，各个Tensor的地址分别为：</p>
           <p>x：起始地址0x0，tensor长度为(4096 * sizeof(float) + 256)字节</p>
           <p>y：起始地址0x4100，tensor长度为(64 * 1024 - (4096 * sizeof(float) + 256))字节</p>
           <p>z：起始地址0x10000，tensor长度为4096 * sizeof(float) 字节</p>
           <p>x多申请256字节，避免一个Repeat内x y同时读同一个bank group；y多申请空间，确保z不会和x/y落入同一个bank</p></td>
         </tr>
         <tr>
          <td>
           <p>示意图</p></td>
          <td>
           <p><span></span></p></td>
          <td>
           <p><span></span></p></td>
         </tr>
         <tr>
          <td>
           <p>示例代码</p></td>
          <td>

             <table>
              <tbody>
               <tr>
                <td>

                  <pre><span>1</span>
<span>2</span>
<span>3</span></pre>
                 </td>
                <td>

                  <pre><span></span><span>pipe</span><span>.</span><span>InitBuffer</span><span>(</span><span>inQueueX</span><span>,</span><span> </span><span>1</span><span>,</span><span> </span><span>4096</span><span> </span><span>*</span><span> </span><span>sizeof</span><span>(</span><span>float</span><span>));</span>
<span>pipe</span><span>.</span><span>InitBuffer</span><span>(</span><span>inQueueY</span><span>,</span><span> </span><span>1</span><span>,</span><span> </span><span>4096</span><span> </span><span>*</span><span> </span><span>sizeof</span><span>(</span><span>float</span><span>));</span>
<span>pipe</span><span>.</span><span>InitBuffer</span><span>(</span><span>outQueueZ</span><span>,</span><span> </span><span>1</span><span>,</span><span> </span><span>4096</span><span> </span><span>*</span><span> </span><span>sizeof</span><span>(</span><span>float</span><span>));</span>
</pre>
                 </td>
               </tr>
              </tbody>
             </table>

           </td>
          <td>

             <table>
              <tbody>
               <tr>
                <td>

                  <pre><span>1</span>
<span>2</span>
<span>3</span></pre>
                 </td>
                <td>

                  <pre><span></span><span>pipe</span><span>.</span><span>InitBuffer</span><span>(</span><span>inQueueX</span><span>,</span><span> </span><span>1</span><span>,</span><span> </span><span>4096</span><span> </span><span>*</span><span> </span><span>sizeof</span><span>(</span><span>float</span><span>)</span><span> </span><span>+</span><span> </span><span>256</span><span>);</span><span> </span><span>// 多申请256字节</span>
<span>pipe</span><span>.</span><span>InitBuffer</span><span>(</span><span>inQueueY</span><span>,</span><span> </span><span>1</span><span>,</span><span> </span><span>64</span><span> </span><span>*</span><span> </span><span>1024</span><span> </span><span>-</span><span> </span><span>(</span><span>4096</span><span> </span><span>*</span><span> </span><span>sizeof</span><span>(</span><span>float</span><span>)</span><span> </span><span>+</span><span> </span><span>256</span><span>));</span><span> </span><span>//多申请空间，确保z不会和x/y落入同一个bank， 64 * 1024是16个bank的空间，4096 * sizeof(float) + 256是x所占的空间</span>
<span>pipe</span><span>.</span><span>InitBuffer</span><span>(</span><span>outQueueZ</span><span>,</span><span> </span><span>1</span><span>,</span><span> </span><span>4096</span><span> </span><span>*</span><span> </span><span>sizeof</span><span>(</span><span>float</span><span>));</span>
</pre>
                 </td>
               </tr>
              </tbody>
             </table>

           </td>
         </tr>
        </tbody>
       </table>

 **父主题：** [避免Unified Buffer的bank冲突](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_0025.html)
