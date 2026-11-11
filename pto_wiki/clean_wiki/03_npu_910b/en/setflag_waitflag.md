> **CANN 9.0.X — local 910B4 / CANN 9.0.0 reading edition.** [Official English page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html).

# SetFlag/WaitFlag(ISASI)

#### Applicability

| Product | Supported |
| --- | --- |
| Atlas 350 Accelerator Card | √ |
| Atlas A3 training product/Atlas A3 inference product | √ |
| Atlas A2 training product/Atlas A2 inference product | √ |
| Atlas 200I/500 A2 inference product | x |
| Atlas inference product AI Core | √ |
| Atlas inference product Vector Core | x |
| Atlas training product | √ |

#### Function Usage

Synchronizes different pipelines in the same core. This synchronization operation needs to be inserted between different pipeline instructions with data dependency.

#### Prototype
```cpp
template <HardEvent event>
__aicore__ inline void SetFlag(int32_t eventID)
template <HardEvent event>
__aicore__ inline void WaitFlag(int32_t eventID)
```

#### Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| event | Input | Template parameter. Synchronization event of the HardEvent data type. For details, see the following description of synchronization types. |
| eventID | Input | Event ID. The data type is int32_t. The definition is as follows: In TPipe- and TQue-based programming scenarios, the event ID needs to be obtained by using **AllocEventID** ([AllocEventID](alloc_event_id.md)) or **FetchEventID** ([FetchEventID](fetch_event_id.md)). Atlas training product: The value ranges from 0 to 3. Atlas inference product AI Core: The value ranges from 0 to 7. Atlas A2 training product/Atlas A2 inference product: The value ranges from 0 to 7. Atlas A3 training product/Atlas A3 inference product: The value ranges from 0 to 7. Atlas 350 Accelerator Card: The value ranges from 0 to 7. |

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

> **Local 2201 events:** The enum above is a generic list, not a supported-event matrix for each core. The installed 2201 implementation excludes `M_V`, `V_M`, `MTE1_V`, and `V_MTE1` from intra-core WaitFlag handling. AIC/AIV data handoff uses GM and cross-core synchronization, rather than those coupled-core events. The installed enum also includes `MTE1_FIX`, `FIX_MTE1`, and `FIX_FIX`; consult its per-core mapping for the selected operation.

#### Returns

None

#### Constraints

- **SetFlag** and **WaitFlag** must be used in pairs.

- In TPipe- and TQue-based programming scenarios, obtain event IDs through **AllocEventID** or **FetchEventID**; do not choose IDs manually. Manually chosen IDs may conflict with framework synchronization events, causing a hang.

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
