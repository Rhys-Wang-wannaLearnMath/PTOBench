> **CANN 9.0.X — local 910B4 / CANN 9.0.0 reading edition.** [Official English page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0114.html).

# AllocEventID

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

Applies for a TEventID of HardEvent (hardware synchronization event). This API must be used together with [ReleaseEventID](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0115.html). After this API is called, the applied TEventID is occupied until **ReleaseEventID** is called.

#### Prototype
```cpp
template <HardEvent evt>
__aicore__ inline TEventID AllocEventID()
```

#### Parameters

| Parameter | Description |
| --- | --- |
| evt | HardEvent hardware synchronization type. For details about this type, see the description of synchronization types in [SetFlag/WaitFlag(ISASI)](setflag_waitflag.md). |

#### Constraints

The number of TEventIDs is limited. After using TEventIDs, call **ReleaseEventID** immediately to release them, preventing TEventIDs from being exhausted.

#### Returns

TEventID

#### Example
```cpp
AscendC::TEventID eventID = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_S>(); //The scalar needs to wait for vector synchronization and the HardEvent ID is required.
AscendC::SetFlag<AscendC::HardEvent::V_S>(eventID);
......
......
......
AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventID);
GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(eventID); //Release the ID of the hard event (scalar waiting for vector synchronization).
......
```
