> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../zh/alloc_event_id.md); [official page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0114.html); [unaltered source body](../raw/9ebc16bfcefbf089.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="EN-US_TOPIC_0000002562301695"></a><a name="EN-US_TOPIC_0000002562301695"></a> 

# AllocEventID

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

Applies for a TEventID of HardEvent (hardware synchronization event). This API must be used together with [ReleaseEventID](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0115.html). After this API is called, the applied TEventID is occupied until **ReleaseEventID** is called.

#### Prototype

```cpp
template <HardEvent evt>
__aicore__ inline TEventID AllocEventID()
```

#### Parameters

<table><b>Table 1 </b>Template parameters<thead><tr><th><p>Parameter</p>
</th>
<th><p>Description</p>
</th>
</tr>
</thead>
<tbody><tr><td><p>evt</p>
</td>
<td><p>HardEvent hardware synchronization type. For details about this type, see the description of synchronization types in <a href="https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html">SetFlag/WaitFlag(ISASI)</a>.</p>
</td>
</tr>
</tbody>
</table>

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

**Parent topic:** [TPipe](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0108.html)
