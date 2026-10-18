> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../en/fetch_event_id.md); [official page](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0116.html); [unaltered source body](../raw/de7765874c53f517.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="ZH-CN_TOPIC_0000002562461857"></a><a name="ZH-CN_TOPIC_0000002562461857"></a> 

# FetchEventID

#### 产品支持情况

<table><thead><tr><th><p><span>产品</span></p>
</th>
<th><p>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr><td><p><span>Atlas 350 加速卡</span></p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p><span>Atlas A3 训练系列产品/Atlas A3 推理系列产品</span></p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p><span>Atlas A2 训练系列产品/Atlas A2 推理系列产品</span></p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p><span>Atlas 200I/500 A2 推理产品</span></p>
</td>
<td><p>x</p>
</td>
</tr>
<tr><td><p><span>Atlas 推理系列产品AI Core</span></p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p><span>Atlas 推理系列产品Vector Core</span></p>
</td>
<td><p>x</p>
</td>
</tr>
<tr><td><p><span>Atlas 训练系列产品</span></p>
</td>
<td><p>√</p>
</td>
</tr>
</tbody>
</table>

#### 功能说明

根据HardEvent（硬件类型的同步事件）获取相应可用的TEventID，此接口不会申请TEventID，仅提供可用的TEventID。

#### 函数原型

```cpp
template <HardEvent evt>
__aicore__ inline TEventID FetchEventID()
__aicore__ inline TEventID FetchEventID(HardEvent evt)
```

#### 参数说明

<table><thead><tr><th><p>参数名</p>
</th>
<th><p>输入/输出</p>
</th>
<th><p>含义</p>
</th>
</tr>
</thead>
<tbody><tr><td><p>evt</p>
</td>
<td><p>输入</p>
</td>
<td><p>HardEvent类型，硬件同步类型。</p>
<p>该类型的具体说明请参考<a href="https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html">SetFlag/WaitFlag(ISASI)</a>中同步类型的说明。</p>
</td>
</tr>
</tbody>
</table>

#### 约束说明

相比于[AllocEventID](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0114.html)，FetchEventID适用于临时使用ID的场景，获取ID后，不会对ID进行占用。在一些复杂的使用场景下，需要开发者自行保证使用正确。比如相同流水连续调用SetFlag/WaitFlag，如果两次传入的ID都是使用FetchEventID获取的，因为两者ID相同会出现程序卡死等未定义行为，这时推荐用户使用AllocEventID。

#### 返回值说明

TEventID

#### 调用示例

```cpp
AscendC::TEventID eventIdVToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S); //需要插scalar等vector的同步，申请对应的HardEvent的ID
AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
```

**父主题：** [TPipe](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0108.html)
