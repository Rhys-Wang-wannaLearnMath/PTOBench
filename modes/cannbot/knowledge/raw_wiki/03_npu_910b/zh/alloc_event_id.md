> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../en/alloc_event_id.md); [official page](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0114.html); [unaltered source body](../raw/efce451fed82365c.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="ZH-CN_TOPIC_0000002562301695"></a><a name="ZH-CN_TOPIC_0000002562301695"></a> 

# AllocEventID

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

用于申请HardEvent（硬件类型同步事件）的TEventID，必须与[ReleaseEventID](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0115.html)搭配使用，调用该接口后，会占用申请的TEventID，直至调用ReleaseEventID释放。

#### 函数原型

```cpp
template <HardEvent evt>
__aicore__ inline TEventID AllocEventID()
```

#### 参数说明

<table><b>表1 </b>模板参数说明<thead><tr><th><p>参数名</p>
</th>
<th><p>描述</p>
</th>
</tr>
</thead>
<tbody><tr><td><p>evt</p>
</td>
<td><p>HardEvent硬件同步类型。该类型的具体说明请参考<a href="https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html">SetFlag/WaitFlag(ISASI)</a>中同步类型的说明。</p>
</td>
</tr>
</tbody>
</table>

#### 约束说明

TEventID有数量限制，使用结束后应该立刻调用ReleaseEventID释放，防止TEventID耗尽。

#### 返回值说明

TEventID

#### 调用示例

```cpp
AscendC::TEventID eventID = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_S>(); //需要插入scalar等vector的同步，申请对应的HardEvent的ID
AscendC::SetFlag<AscendC::HardEvent::V_S>(eventID);
......
......
......
AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventID);
GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(eventID); //释放scalar等vector的同步HardEvent的ID
......
```

**父主题：** [TPipe](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0108.html)
