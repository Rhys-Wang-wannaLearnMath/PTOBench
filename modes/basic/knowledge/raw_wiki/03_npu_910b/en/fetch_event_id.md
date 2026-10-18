> **Official CANN 9.0.X source reading copy.** [English/Chinese counterpart](../zh/fetch_event_id.md); [official page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0116.html); [unaltered source body](../raw/d49e7a2cca51e5ea.html). HTML formatting and image paths are converted for offline reading; no local technical corrections.

<a name="EN-US_TOPIC_0000002562461857"></a><a name="EN-US_TOPIC_0000002562461857"></a> 

# FetchEventID

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

Obtains available TEventIDs based on HardEvent (hardware synchronization event). This API does not allocate TEventIDs, but provides available TEventIDs.

#### Prototype

```cpp
template <HardEvent evt>
__aicore__ inline TEventID FetchEventID()
__aicore__ inline TEventID FetchEventID(HardEvent evt)
```

#### Parameters

<table><thead><tr><th><p>Parameter</p>
</th>
<th><p>Input/Output</p>
</th>
<th><p>Meaning</p>
</th>
</tr>
</thead>
<tbody><tr><td><p>evt</p>
</td>
<td><p>Input</p>
</td>
<td><p>HardEvent hardware synchronization type.</p>
<p>For details about this type, see the description of synchronization types in <a href="https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html">SetFlag/WaitFlag(ISASI)</a>.</p>
</td>
</tr>
</tbody>
</table>

#### Constraints

Compared with [AllocEventID](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0114.html), **FetchEventID** is suitable for scenarios requiring temporary ID usage, as it does not occupy the ID after acquisition. In some complex application scenarios, you need to ensure that the usage is correct. For example, if **SetFlag** and **WaitFlag** are called consecutively in the same pipeline and the two IDs are obtained using **FetchEventID**, undefined behaviors such as program suspension may occur because the two IDs are the same. In this case, **AllocEventID** is recommended.

#### Returns

TEventID

#### Example

```cpp
AscendC::TEventID eventIdVToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S); //The scalar needs to wait for vector synchronization and the HardEvent ID is required.
AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
```

**Parent topic:** [TPipe](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0108.html)
