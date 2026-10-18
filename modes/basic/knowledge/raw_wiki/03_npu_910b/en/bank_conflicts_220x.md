> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../zh/bank_conflicts_220x.md); [official page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00020.html); [unaltered source body](../raw/95197eda21dbafe1.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="EN-US_TOPIC_0000002531509894"></a><a name="EN-US_TOPIC_0000002531509894"></a> 

# Avoiding Bank Conflicts (NPU Architecture Version 220x)

[Priority] High

![](../raw/95ff0a306791d19e.png)

This performance optimization suggestion applies to the following product models:

- Atlas A3 training product / Atlas A3 inference product

- Atlas A2 training product / Atlas A2 inference product

[Description] To improve data access efficiency and throughput, the Unified Buffer adopts a bank (memory modules of the same size) structure. The Unified Buffer has a total size of 192 KB and is divided into 48 banks. Each bank consists of 128 rows, with each row being 32 bytes long. These 48 banks are further organized into 16 bank groups, with each bank group containing three banks. For example, bank15, bank31, and bank47 form a bank group.

 **Figure 1 **Bank structure (the arrow direction indicates the memory layout sequence) 

![](../raw/58b83f4596d5e804.png)

Each bank can independently read and write data, and multiple data requests can be processed at the same time. However, when a plurality of read/write operations attempt to access a same bank or bank group at the same time, due to a limitation of hardware resources, these operations need to wait in a queue, resulting in a bank conflict and performance deterioration.

Specifically, the Vector Unit can read or write a row of data from or into each bank group in each cycle (an instruction cycle). If multiple operations in the same API attempt to access the same bank or bank group at the same time, the Vector Unit cannot process all requests in the same cycle. As a result, these requests wait in a queue. This queuing increases the data access latency and reduces the overall system performance.

#### Typical Scenarios of Bank Conflicts

Bank conflicts can be classified into the following three scenarios:

- **Read/Write conflict**: A read operation and a write operation attempt to access the same bank at the same time.

- **Write/Write conflict**: Multiple write operations attempt to access the same bank group at the same time.

- **Read/Read conflict**: Multiple read operations attempt to access the same bank group at the same time.

The following provides some specific examples. Assume that 0x10000 is on bank16, 0x10020 is on bank17, and 0x20020 is on bank33, as shown in the following figure.

 **Figure 2 **Address allocation 

![](../raw/90833a47249e7a75.png)

- Example of a read/write conflict 

 <a name="EN-US_TOPIC_0000002531509894__li1987054183914"></a><a name="li1987054183914"></a>When the source operand src and destination operand dst of the Vector instruction are read and written to the same bank at the same time, a read/write conflict occurs. The details are as follows: 

<table>

          <b>Table 1 </b>Example of a read/write conflict

         <thead>
          <tr>
           <th>
            <p>No.</p></th>
           <th>
            <p>src addr</p></th>
           <th>
            <p>dst addr</p></th>
           <th>
            <p>bank</p></th>
           <th>
            <p>bank group</p></th>
           <th>
            <p>Conclusion</p></th>
          </tr>
         </thead>
         <tbody>
          <tr>
           <td>
            <p>Example 1</p></td>
           <td>
            <p>0x10020</p></td>
           <td>
            <p>0x10000</p></td>
           <td>
            <p>bank_id0 != bank_id1</p></td>
           <td>
            <p>bank_group_id0 != bank_group_id1</p></td>
           <td>
            <p>The addresses of src and dst are in bank16 and bank17, respectively. Therefore, no conflict occurs.</p></td>
          </tr>
          <tr>
           <td>
            <p>Example 2</p></td>
           <td>
            <p>0x10020</p></td>
           <td>
            <p>0x10E20</p></td>
           <td>
            <p>bank_id0 == bank_id1</p></td>
           <td>
            <p>bank_group_id0 == bank_group_id1</p></td>
           <td>
            <p>The addresses of src and dst are both in bank17. Therefore, a conflict occurs.</p></td>
          </tr>
         </tbody>
        </table>

- Example of a write/write conflict 

 <a name="EN-US_TOPIC_0000002531509894__li1264151618456"></a><a name="li1264151618456"></a>When eight data blocks (block0 to block7) corresponding to the destination operand dst of the Vector instruction are written to the same bank group, a write/write conflict occurs. The details are as follows: 

<table>

          <b>Table 2 </b>Example of a write/write conflict

         <thead>
          <tr>
           <th>
            <p>No.</p></th>
           <th>
            <p>dst addr</p></th>
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
            <p>Conclusion</p></th>
          </tr>
         </thead>
         <tbody>
          <tr>
           <td>
            <p>Example 1</p></td>
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
            <p>All eight data blocks are in the same bank group. Therefore, conflicts occur. One repeat is written in eight cycles.</p></td>
          </tr>
          <tr>
           <td>
            <p>Example 2</p></td>
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
            <p>block0 and block2 are in the same bank group and conflict with each other. One repeat is written in four cycles.</p></td>
          </tr>
         </tbody>
        </table>

- Read/Read conflict 

- When multiple source operands of the Vector instruction are read to the same bank group at the same time, a read/read conflict occurs. The analysis is as follows: 

<table>

           <b>Table 3 </b>Example of a read/read conflict with two source addresses

          <thead>
           <tr>
            <th>
             <p>No.</p></th>
            <th>
             <p>src0 addr</p></th>
            <th>
             <p>src1 addr</p></th>
            <th>
             <p>bank</p></th>
            <th>
             <p>bank group</p></th>
            <th>
             <p>Conclusion</p></th>
           </tr>
          </thead>
          <tbody>
           <tr>
            <td>
             <p>Example 1</p></td>
            <td>
             <p>0x10020</p></td>
            <td>
             <p>0x20020</p></td>
            <td>
             <p>bank_id0 != bank_id1</p></td>
            <td>
             <p>bank_group_id0 == bank_group_id1</p></td>
            <td>
             <p>Conflict</p></td>
           </tr>
           <tr>
            <td>
             <p>Example 2</p></td>
            <td>
             <p>0x10020</p></td>
            <td>
             <p>0x10000</p></td>
            <td>
             <p>bank_id0 != bank_id1</p></td>
            <td>
             <p>bank_group_id0 != bank_group_id1</p></td>
            <td>
             <p>No conflict</p></td>
           </tr>
          </tbody>
         </table>

- When the eight data blocks (block0 to block7) corresponding to a source operand of the Vector instruction are read to the same bank group, a read/read conflict occurs. The analysis is as follows: 

<table>

           <b>Table 4 </b>Example of a read/read conflict with a single source address

          <thead>
           <tr>
            <th>
             <p>No.</p></th>
            <th>
             <p>src addr</p></th>
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
             <p>Conclusion</p></th>
           </tr>
          </thead>
          <tbody>
           <tr>
            <td>
             <p>Example 1</p></td>
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
             <p>All eight data blocks are in the same bank group. Therefore, conflicts occur. One repeat is read in eight cycles.</p></td>
           </tr>
           <tr>
            <td>
             <p>Example 2</p></td>
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
             <p>block0 and block2 are in the same bank group. Therefore, a conflict occurs. One repeat is completed in four cycles.</p></td>
           </tr>
          </tbody>
         </table>

![](../raw/95ff0a306791d19e.png)

The msProf tool can be used to collect profile data related to the resource conflict ratio.

For details about how to use the tool and the description of the profile data file, see [Operator Development Tools](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/optool/docs/en/quick_start/mskpp_quick_start.md).

#### How to Avoid Bank Conflicts

There are two methods to avoid bank conflicts: **optimizing the computation logic** and **optimizing address allocation**.

- **Optimizing the computation logic** 

Perform the (1, 0, 2) transpose operation on an input with shape (8, 16, 16). Then, the output shape is (16, 8, 16). You can avoid conflicts by changing the computation logic from "skipped read and continuous write" to "continuous read and skipped write". The implementation solutions are compared as follows:

<table>
        <thead>
         <tr>
          <th>
           <p>Implementation</p></th>
          <th>
           <p>Original Implementation</p></th>
          <th>
           <p>Optimized Implementation</p></th>
         </tr>
        </thead>
        <tbody>
         <tr>
          <td>
           <p>Implementation method</p></td>
          <td>
           <p>Read skipping, continuous write</p>
           <p>The eight data blocks input in the same repeat are in the same bank group, causing a read/read conflict.</p></td>
          <td>
           <p>Continuous read, skip write</p>
           <p>The eight data blocks input in the same repeat are not in the same bank group, avoiding a read/read conflict.</p></td>
         </tr>
         <tr>
          <td>
           <p>Diagram</p></td>
          <td>
           <p><span></span></p></td>
          <td>
           <p><span></span></p></td>
         </tr>
         <tr>
          <td>
           <p>Sample code</p></td>
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

- **Optimizing address allocation** 

Implement the addition of 4096 consecutive float elements (z = x + y). By allocating more memory, ensure that x and y do not appear in the same bank group at the same time, and x/y and z do not appear in the same bank at the same time within a repeat. For details about the complete example, see [sample of avoiding bank conflicts](https://gitee.com/ascend/samples/tree/master/operator/ascendc/4_best_practices/4_bank_conflict).

The implementation solutions are compared as follows:

<table>
        <thead>
         <tr>
          <th>
           <p>Implementation</p></th>
          <th>
           <p>Original Implementation</p></th>
          <th>
           <p>Optimized Implementation</p></th>
         </tr>
        </thead>
        <tbody>
         <tr>
          <td>
           <p>Implementation method</p></td>
          <td>
           <p>No address optimization is performed. InitBuffer is directly used to allocate memory. The addresses of each tensor are as follows:</p>
           <p>x: The start address is 0x0, and the tensor length is 4096 x sizeof(float) bytes.</p>
           <p>y: The start address is 0x4000, and the tensor length is 4096 x sizeof(float) bytes.</p>
           <p>z: The start address is 0x8000, and the tensor length is 4096 x sizeof(float) bytes.</p>
           <p>In a repeat, x and y read the same bank group at the same time, and x/y and z read and write the same bank at the same time.</p></td>
          <td>
           <p>Optimize the address. When InitBuffer is used to allocate memory, properly allocate more memory. The addresses of each tensor are as follows:</p>
           <p>x: The start address is 0x0, and the tensor length is (4096 x sizeof(float) + 256) bytes.</p>
           <p>y: The start address is 0x4100, and the tensor length is (64 x 1024 – (4096 x sizeof(float) + 256)) bytes.</p>
           <p>z: The start address is 0x10000, and the tensor length is 4096 x sizeof(float) bytes.</p>
           <p>Allocate 256 bytes more for x to prevent x and y from reading the same bank group in a repeat. Allocate more space for y to ensure that z does not fall into the same bank as x or y.</p></td>
         </tr>
         <tr>
          <td>
           <p>Diagram</p></td>
          <td>
           <p><span></span></p></td>
          <td>
           <p><span></span></p></td>
         </tr>
         <tr>
          <td>
           <p>Sample code</p></td>
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

                  <pre><span></span><span>pipe</span><span>.</span><span>InitBuffer</span><span>(</span><span>inQueueX</span><span>,</span><span> </span><span>1</span><span>,</span><span> </span><span>4096</span><span> </span><span>*</span><span> </span><span>sizeof</span><span>(</span><span>float</span><span>)</span><span> </span><span>+</span><span> </span><span>256</span><span>);</span><span> </span><span>// Allocate 256 bytes more.</span>
<span>pipe</span><span>.</span><span>InitBuffer</span><span>(</span><span>inQueueY</span><span>,</span><span> </span><span>1</span><span>,</span><span> </span><span>64</span><span> </span><span>*</span><span> </span><span>1024</span><span> </span><span>-</span><span> </span><span>(</span><span>4096</span><span> </span><span>*</span><span> </span><span>sizeof</span><span>(</span><span>float</span><span>)</span><span> </span><span>+</span><span> </span><span>256</span><span>));</span><span> </span><span>// Allocate more space to ensure that z does not fall into the same bank as x/y. 64 * 1024 is the space of 16 banks, and 4096 * sizeof(float) + 256 is the space occupied by x.</span>
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

 **Parent topic:** [Avoiding Bank Conflicts in the Unified Buffer](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_0025.html)
