> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../zh/setflag_waitflag.md); [official page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html); [unaltered source body](../raw/4aa1ffc57df8bdd8.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="EN-US_TOPIC_0000002531541350"></a><a name="EN-US_TOPIC_0000002531541350"></a> 

# SetFlag/WaitFlag(ISASI)

#### Applicability

<table><thead><tr><th><p><span>Product</span></p>
</th>
<th><p>Supported</p>
</th>
</tr>
</thead>
<tbody><tr><td><p><span>Atlas 350 Accelerator Card</span></p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p><span>Atlas A3 training product/Atlas A3 inference product</span></p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p><span>Atlas A2 training product/Atlas A2 inference product</span></p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p><span>Atlas 200I/500 A2 inference product</span></p>
</td>
<td><p>x</p>
</td>
</tr>
<tr><td><p><span>Atlas inference product AI Core</span></p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p><span>Atlas inference product Vector Core</span></p>
</td>
<td><p>x</p>
</td>
</tr>
<tr><td><p><span>Atlas training product</span></p>
</td>
<td><p>√</p>
</td>
</tr>
</tbody>
</table>

#### Function Usage

Synchronizes different pipelines in the same core. This synchronization operation needs to be inserted between different pipeline instructions with data dependency.

#### Prototype

```cpp
template <HardEvent event>
__aicore__ inline void SetFlag(int32_t eventID)
template <HardEvent event>
__aicore__ inline void WaitFlag(int32_t eventID)
```

<a name="EN-US_TOPIC_0000002531541350__section622mcpsimp"></a><a name="section622mcpsimp"></a>

#### Parameters

<table><b>Table 1 </b>Parameters<thead><tr><th><p>Parameter</p>
</th>
<th><p>Input/Output</p>
</th>
<th><p>Description</p>
</th>
</tr>
</thead>
<tbody><tr><td><p>event</p>
</td>
<td><p>Input</p>
</td>
<td><p>Template parameter.</p>
<p>Synchronization event of the HardEvent data type. For details, see the following description of synchronization types.</p>
</td>
</tr>
<tr><td><p>eventID</p>
</td>
<td><p>Input</p>
</td>
<td><p>Event ID. The data type is int32_t. The definition is as follows:</p>
<p>In TPipe- and TQue-based programming scenarios, the event ID needs to be obtained by using <strong>AllocEventID</strong> (<a href="https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0114.html">AllocEventID</a>) or <strong>FetchEventID</strong> (<a href="https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0116.html">FetchEventID</a>).</p>
<p><span>Atlas training product</span>: The value ranges from 0 to 3.</p>
<p><span>Atlas inference product AI Core</span>: The value ranges from 0 to 7.</p>
<p><span>Atlas A2 training product/Atlas A2 inference product</span>: The value ranges from 0 to 7.</p>
<p><span>Atlas A3 training product/Atlas A3 inference product</span>: The value ranges from 0 to 7.</p>
<p><span>Atlas 350 Accelerator Card</span>: The value ranges from 0 to 7.</p>
</td>
</tr>
</tbody>
</table>

The synchronization types are described as follows:

```cpp
enum class HardEvent : uint8_t {
   // Name (source pipeline_target pipeline). For example, MTE2_V indicates that PIPE_MTE2 is the source pipeline and PIPE_V is the target pipeline. Synchronization is performed between PIPE_MTE2 and PIPE_V, and PIPE_V waits for PIPE_MTE2.
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

#### Returns

None

#### Constraints

- **SetFlag** and **WaitFlag** must be used in pairs.

- In TPipe- and TQue-based programming scenarios, you are not advised to specify the event ID when using **SetFlag** and **WaitFlag**. Otherwise, the event ID may conflict with the framework synchronization event, causing suspension.

- In static tensor programming scenarios, the event types and event IDs are managed by developers. However, note that event IDs 6 and 7 cannot be used (as they may conflict with internal event IDs, leading to undefined behavior).

#### Example

**DataCopy** can be executed only after **SetValue** is executed. In this case, you need to insert the synchronization operation between **PIPE_S** and **PIPE_MTE3**.

```cpp
AscendC::GlobalTensor<half> dstGlobal;
AscendC::LocalTensor<half> dstLocal;
dstLocal.SetValue(0, 0);
uint32_t dataSize = 512;
// In TPipe- and TQue-based programming scenarios, the event ID needs to be obtained by using AllocEventID or FetchEventID.
int32_t eventIDSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
// In static tensor programming scenarios, the event ID is managed by developers.
// AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
// AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
AscendC::DataCopy(dstGlobal, dstLocal, dataSize);
```

**Parent topic:** [Intra-Core Synchronization](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00013.html)
