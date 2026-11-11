> **CANN 9.0.X — local 910B4 / CANN 9.0.0 reading edition.** Official English page (`https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0116.html`).

# FetchEventID

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

Obtains available TEventIDs based on HardEvent (hardware synchronization event). This API does not allocate TEventIDs, but provides available TEventIDs.

#### Prototype
```cpp
template <HardEvent evt>
__aicore__ inline TEventID FetchEventID()
__aicore__ inline TEventID FetchEventID(HardEvent evt)
```

#### Parameters

| Parameter | Input/Output | Meaning |
| --- | --- | --- |
| evt | Input | HardEvent hardware synchronization type. For details about this type, see the description of synchronization types in [SetFlag/WaitFlag(ISASI)](source-03_npu_910b--en--setflag_waitflag.md). |

#### Constraints

Compared with [AllocEventID](source-03_npu_910b--en--alloc_event_id.md), **FetchEventID** is suitable for scenarios requiring temporary ID usage, as it does not occupy the ID after acquisition. In some complex application scenarios, you need to ensure that the usage is correct. For example, two live synchronization operations of the same HardEvent may obtain the same ID through **FetchEventID** before their matching waits complete. Reusing that ID while it is still pending can cause undefined behavior such as a hang; use **AllocEventID** to reserve separate IDs. A single matched **SetFlag/WaitFlag** pair must use the same ID, as in the example below.

#### Returns

TEventID

#### Example
```cpp
AscendC::TEventID eventIdVToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S); //The scalar needs to wait for vector synchronization and the HardEvent ID is required.
AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
```
