> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../en/setflag_waitflag.md); [official page](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html); [unaltered source body](../raw/1291d7ad59cffb01.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="ZH-CN_TOPIC_0000002531541350"></a><a name="ZH-CN_TOPIC_0000002531541350"></a> 

# SetFlag/WaitFlag(ISASI)

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

同一核内不同流水之间的同步指令。具有数据依赖的不同流水指令之间需要插此同步。

#### 函数原型

```cpp
template <HardEvent event>
__aicore__ inline void SetFlag(int32_t eventID)
template <HardEvent event>
__aicore__ inline void WaitFlag(int32_t eventID)
```

<a name="ZH-CN_TOPIC_0000002531541350__section622mcpsimp"></a><a name="section622mcpsimp"></a>

#### 参数说明

<table><b>表1 </b>参数说明<thead><tr><th><p>参数名</p>
</th>
<th><p>输入/输出</p>
</th>
<th><p>描述</p>
</th>
</tr>
</thead>
<tbody><tr><td><p>event</p>
</td>
<td><p>输入</p>
</td>
<td><p>模板参数。</p>
<p>同步事件，数据类型为HardEvent。详细内容参考下文中的同步类型说明。</p>
</td>
</tr>
<tr><td><p>eventID</p>
</td>
<td><p>输入</p>
</td>
<td><p>事件ID。数据类型为int32_t类型。其定义如下：</p>
<p>在基于TPipe和TQue编程场景中，eventID需要通过<a href="https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0114.html">AllocEventID</a>或者<a href="https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0116.html">FetchEventID</a>来获取。</p>
<p><span>Atlas 训练系列产品</span>，数据范围为：0-3</p>
<p><span>Atlas 推理系列产品AI Core</span>，数据范围为：0-7</p>
<p><span>Atlas A2 训练系列产品/Atlas A2 推理系列产品</span>，数据范围为：0-7</p>
<p><span>Atlas A3 训练系列产品/Atlas A3 推理系列产品</span>，数据范围为：0-7</p>
<p><span>Atlas 350 加速卡</span>，数据范围为：0-7</p>
</td>
</tr>
</tbody>
</table>

同步类型说明如下:

```cpp
enum class HardEvent : uint8_t {
    // 名称（源流水_目标流水），例如MTE2_V，代表PIPE_MTE2为源流水，PIPE_V为目标流水。标识从PIPE_MTE2到PIPE_V的同步，PIPE_V等待PIPE_MTE2。
    MTE2_MTE1
    MTE1_MTE2
    MTE1_M
    M_MTE1
    MTE2_V
    V_MTE2
    MTE3_V
    V_MTE3
    M_V
    V_M
    V_V
    MTE3_MTE1
    MTE1_MTE3
    MTE1_V
    MTE2_M
    M_MTE2
    V_MTE1
    M_FIX
    FIX_M
    MTE3_MTE2
    MTE2_MTE3
    S_V
    V_S
    S_MTE2
    MTE2_S
    S_MTE3
    MTE3_S
    MTE2_FIX
    FIX_MTE2
    FIX_S
    M_S
    FIX_MTE3
}
```

#### 返回值说明

无

#### 约束说明

- SetFlag/WaitFlag必须成对出现。

- 在基于TPipe和TQue编程场景中，禁止用户在使用SetFlag和WaitFlag时自行指定eventID，容易与框架同步事件冲突，导致卡死问题。

- 在静态Tensor编程场景中，事件的类型和事件ID由开发者自行管理，但需要注意事件ID不能使用6和7（可能与内部使用的事件ID出现冲突，进而出现未定义行为）。

#### 调用示例

如DataCopy需要等待SetValue执行完成后才能执行，需要插入PIPE_S到PIPE_MTE3的同步。

```cpp
AscendC::GlobalTensor<half> dstGlobal;
AscendC::LocalTensor<half> dstLocal;
dstLocal.SetValue(0, 0);
uint32_t dataSize = 512;
// 基于TPipe和TQue编程场景中，eventID需要通过AllocEventID或FetchEventID获取
int32_t eventIDSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
// 静态Tensor编程场景中，eventID由开发者自行管理
// AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
// AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
AscendC::DataCopy(dstGlobal, dstLocal, dataSize);
```

**父主题：** [核内同步](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00013.html)
