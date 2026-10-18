<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="search_title" content=""></head><body><a name="EN-US_TOPIC_0000002660762923"></a><a name="EN-US_TOPIC_0000002660762923"></a>

<h1 class="topictitle1">CANN</h1>
<div id="body39451090"><p id="shortdesc1991014334266">Compute Architecture for Neural Networks (CANN) is designed for AI tasks on Ascend hardware. It supports AI frameworks such as MindSpore, PyTorch, and TensorFlow. Boosting the performance of AI processors and programming, CANN offers ready-to-use APIs for different uses, allowing you to quickly create AI applications and services with Ascend.</p>
<div class="section" id="EN-US_TOPIC_0000002660762923__section940560596"><h4 class="sectiontitle">Key Features of CANN 9.1.0</h4><ul id="EN-US_TOPIC_0000002660762923__ul196716535513"><li id="EN-US_TOPIC_0000002660762923__li96795318512">Added support for <span id="EN-US_TOPIC_0000002660762923__ph16919143314559"><term id="EN-US_TOPIC_0000002660762923__en-us_topic_0000002630551350_term2013610186597">Ascend 950DT</term></span>, providing higher bandwidth in training and decoding inference tasks.</li><li id="EN-US_TOPIC_0000002660762923__li367175320520">Added core operators such as SMLA and mHC based on Atlas A2, Atlas A3, and Ascend 950 products. Adapted to open-source models such as DeepSeek V4, Qwen 3.6, Kimi-K2.6, and GLM 5.2.</li><li id="EN-US_TOPIC_0000002660762923__li166717535517">Released operator development documentation for beginners, intermediate users, and advanced developers. (<a href="https://www.hiascend.com/en/developer/operator?tab=ascendc" target="_blank" rel="noopener noreferrer">Details</a>)</li><li id="EN-US_TOPIC_0000002660762923__li167195317517">Based on Ascend 950 products, SHMEM now supports AI Cores directly issuing MTE/UDMA/RDMA data transfers, enabling UB and RoCE communication, and provides matching programming APIs. (<a href="https://gitcode.com/cann/shmem/blob/master/README.md" target="_blank" rel="noopener noreferrer">Details</a>)</li><li id="EN-US_TOPIC_0000002660762923__li18671453552">Based on Ascend 950 products, CATLASS now supports Tile components and provides a full set of MxFP8/MxFP4 quantization templates and sample libraries. (<a href="https://gitcode.com/cann/catlass/blob/master/README.md" target="_blank" rel="noopener noreferrer">Details</a>)</li><li id="EN-US_TOPIC_0000002660762923__li206711532054">The operator precision standard is open-sourced. The precision of floating-point compute operators can now be determined using hybrid tolerance metrics. (<a href="https://gitcode.com/cann/opbase/blob/9.1.0/docs/en/ops_precision_standard/experimental_standard.md" target="_blank" rel="noopener noreferrer">Details</a>)</li></ul>
<p id="EN-US_TOPIC_0000002660762923__p12949183014544"></p>
</div>
<div class="section" id="EN-US_TOPIC_0000002660762923__section1616513217166"><h4 class="sectiontitle">Quick Start</h4><p id="EN-US_TOPIC_0000002660762923__p15453164513111"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/softwareinst/releasenote/9.1.0/release-notes.md" target="_blank" rel="noopener noreferrer">Release Notes</a></p>
<p id="EN-US_TOPIC_0000002660762923__p1413722881611"><a href="https://www.hiascend.com/en/cann/download?versionId=791&amp;ids=d806,h0501,h0601,h0703" target="_blank" rel="noopener noreferrer">Quick Installation</a></p>
<p id="EN-US_TOPIC_0000002660762923__p171081842101612"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/guide/入门教程/Ascend-C概述与学习路径.md" target="_blank" rel="noopener noreferrer">Ascend C Programming Guide</a></p>
</div>
<div class="section" id="EN-US_TOPIC_0000002660762923__section56001751163915"><h4 class="sectiontitle">CANN Documentation</h4><p id="EN-US_TOPIC_0000002660762923__p9337125963919">View CANN documents by workflow, function, API, and more.</p>
</div>

<div class="tablenoborder"><table cellpadding="4" cellspacing="0" summary="" id="EN-US_TOPIC_0000002660762923__table88931501501" frame="border" border="1" rules="all"><caption><b>Table 1 </b>Release and Installation</caption><thead align="left"><tr id="EN-US_TOPIC_0000002660762923__row98931901010"><th align="left" class="cellrowborder" valign="top" width="100%" id="mcps1.4.4.2.2.1.1"><p id="EN-US_TOPIC_0000002660762923__p12893301015">Release and Installation</p>
</th>
</tr>
</thead>
<tbody><tr id="EN-US_TOPIC_0000002660762923__row198931807010"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.4.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p158937017010"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/softwareinst/releasenote/9.1.0/release-notes.md" target="_blank" rel="noopener noreferrer">Release Notes</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row1489310604"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.4.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1889319020015"><a href="https://www.hiascend.com/en/cann/download" target="_blank" rel="noopener noreferrer">Quick Installation</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row0893104016"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.4.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p17893301007"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html" target="_blank" rel="noopener noreferrer">Software Installation</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row1089360605"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.4.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p18931805014"><a href="https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html" target="_blank" rel="noopener noreferrer">Ascend Product Models</a></p>
</td>
</tr>
</tbody>
</table>
</div>

<div class="tablenoborder"><table cellpadding="4" cellspacing="0" summary="" id="EN-US_TOPIC_0000002660762923__table157017301345" frame="border" border="1" rules="all"><caption><b>Table 2 </b>Learning and Tutorials</caption><thead align="left"><tr id="EN-US_TOPIC_0000002660762923__row11570530143413"><th align="left" class="cellrowborder" valign="top" width="100%" id="mcps1.4.5.2.2.1.1"><p id="EN-US_TOPIC_0000002660762923__p1174023143514">Learning and Tutorials</p>
</th>
</tr>
</thead>
<tbody><tr id="EN-US_TOPIC_0000002660762923__row1957017303344"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.5.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p6570163083414"><a href="https://atomgit.com/cann/cann-learning-hub" target="_blank" rel="noopener noreferrer">CANN-learning-hub</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row8570123063410"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.5.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1057017302340"><a href="https://atomgit.com/cann/cann-samples" target="_blank" rel="noopener noreferrer">CANN-samples</a></p>
</td>
</tr>
</tbody>
</table>
</div>

<div class="tablenoborder"><table cellpadding="4" cellspacing="0" summary="" id="EN-US_TOPIC_0000002660762923__table1613162611484" frame="border" border="1" rules="all"><caption><b>Table 3 </b>Programming Languages</caption><thead align="left"><tr id="EN-US_TOPIC_0000002660762923__row8131202684815"><th align="left" class="cellrowborder" valign="top" width="100%" id="mcps1.4.6.2.2.1.1"><p id="EN-US_TOPIC_0000002660762923__p1813212611486">Programming Language</p>
</th>
</tr>
</thead>
<tbody><tr id="EN-US_TOPIC_0000002660762923__row9132142644818"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.6.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p129155414313"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/guide/入门教程/Ascend-C概述与学习路径.md" target="_blank" rel="noopener noreferrer">Ascend C Programming Guide</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row16132102624817"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.6.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1613213263483"><a href="https://gitcode.com/cann/catlass/blob/master/docs/en/1_Practice/10_innovative_example_development_guide.md" target="_blank" rel="noopener noreferrer">CATLASS Programming Guide</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row1513232654819"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.6.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p31321926204820"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/pyptoopdevg/docs/en/tutorials/introduction/introduction.md" target="_blank" rel="noopener noreferrer">PyPTO Operator Development</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row917762683313"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.6.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p2177626103317"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/graphdevg/atlasag_25_0081.html" target="_blank" rel="noopener noreferrer">Graph Development</a></p>
</td>
</tr>
</tbody>
</table>
</div>

<div class="tablenoborder"><table cellpadding="4" cellspacing="0" summary="" id="EN-US_TOPIC_0000002660762923__table48120491648" frame="border" border="1" rules="all"><caption><b>Table 4 </b>Compiler</caption><thead align="left"><tr id="EN-US_TOPIC_0000002660762923__row181649548"><th align="left" class="cellrowborder" valign="top" width="100%" id="mcps1.4.7.2.2.1.1"><p id="EN-US_TOPIC_0000002660762923__p98120491412">Compiler</p>
</th>
</tr>
</thead>
<tbody><tr id="EN-US_TOPIC_0000002660762923__row198124919410"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.7.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p5818491641"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/compiler/BishengCompiler/atlas_bisheng_10_0001.html" target="_blank" rel="noopener noreferrer">BiSheng Compiler</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row116941616203419"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.7.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1861274115348"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/compiler/ptoisa/docs/PTO-ISA-Header-and-Library-Description.md" target="_blank" rel="noopener noreferrer">PTO Virtual Instruction Set</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row1083917181344"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.7.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p156131441103412"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/compiler/AscendNPUIR/ir_001.html" target="_blank" rel="noopener noreferrer">AscendNPU IR</a></p>
</td>
</tr>
</tbody>
</table>
</div>

<div class="tablenoborder"><table cellpadding="4" cellspacing="0" summary="" id="EN-US_TOPIC_0000002660762923__table184119142476" frame="border" border="1" rules="all"><caption><b>Table 5 </b>API references</caption><thead align="left"><tr id="EN-US_TOPIC_0000002660762923__row194115141476"><th align="left" class="cellrowborder" valign="top" width="100%" id="mcps1.4.8.2.2.1.1"><p id="EN-US_TOPIC_0000002660762923__p17411121484718">API Reference</p>
</th>
</tr>
</thead>
<tbody><tr id="EN-US_TOPIC_0000002660762923__row63353587620"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.8.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p147673312714"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/headerliblist/hfandlf_09_0001.html" target="_blank" rel="noopener noreferrer">Header and Library Files</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row141161464713"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.8.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1441181434718"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/atlasascendc_api_07_0003.html" target="_blank" rel="noopener noreferrer">Ascend C APIs</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row13411171413472"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.8.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p96021459151"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/pyptoapi/docs/en/api/tensor/pypto-Tensor_introduction.md" target="_blank" rel="noopener noreferrer">PyPTO APIs</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row237010541519"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.8.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1941117144478"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/operatorlist_00001.html" target="_blank" rel="noopener noreferrer">Operator Library APIs</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row186021359250"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.8.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p11331118193512"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0019.xml" target="_blank" rel="noopener noreferrer">Runtime APIs</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row176413571058"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.8.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p965155718513"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendgraphapi/atlasgeapi_07_0001.html" target="_blank" rel="noopener noreferrer">GE APIs</a></p>
</td>
</tr>
</tbody>
</table>
</div>

<div class="tablenoborder"><table cellpadding="4" cellspacing="0" summary="" id="EN-US_TOPIC_0000002660762923__table1510152311711" frame="border" border="1" rules="all"><caption><b>Table 6 </b>Operator libraries</caption><thead align="left"><tr id="EN-US_TOPIC_0000002660762923__row18101231972"><th align="left" class="cellrowborder" valign="top" width="100%" id="mcps1.4.9.2.2.1.1"><p id="EN-US_TOPIC_0000002660762923__p1510192311713">Operator Library</p>
</th>
</tr>
</thead>
<tbody><tr id="EN-US_TOPIC_0000002660762923__row81042316713"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.9.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p710823475"><a href="https://gitcode.com/cann/ops-math/tree/9.1.0/docs" target="_blank" rel="noopener noreferrer">ops-math</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row191771346171"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.9.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p141778461176"><a href="https://gitcode.com/cann/ops-cv/tree/9.1.0/docs" target="_blank" rel="noopener noreferrer">ops-cv</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row181791148771"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.9.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p131799487717"><a href="https://gitcode.com/cann/ops-nn/tree/9.1.0/docs" target="_blank" rel="noopener noreferrer">ops-nn</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row12301950675"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.9.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1723015508712"><a href="https://gitcode.com/cann/ops-transformer/tree/9.1.0/docs" target="_blank" rel="noopener noreferrer">ops-transformer</a></p>
</td>
</tr>
</tbody>
</table>
</div>

<div class="tablenoborder"><table cellpadding="4" cellspacing="0" summary="" id="EN-US_TOPIC_0000002660762923__table19666163514169" frame="border" border="1" rules="all"><caption><b>Table 7 </b>Communication libraries</caption><thead align="left"><tr id="EN-US_TOPIC_0000002660762923__row16661935131619"><th align="left" class="cellrowborder" valign="top" width="100%" id="mcps1.4.10.2.2.1.1"><p id="EN-US_TOPIC_0000002660762923__p116665357165">Communication Library</p>
</th>
</tr>
</thead>
<tbody><tr id="EN-US_TOPIC_0000002660762923__row16666835101612"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.10.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1666611355162"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_intro.md" target="_blank" rel="noopener noreferrer">Huawei Collective Communication Library (HCCL)</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row20666135141612"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.10.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p116661935101613"><a href="https://shmem-doc.pages.dev/" target="_blank" rel="noopener noreferrer">SHMEM</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row46662035151620"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.10.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1343264133816"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hixlug/hixldev_001.html" target="_blank" rel="noopener noreferrer">HIXL Unilateral Communication Library</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row11666123510167"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.10.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p16666173518167"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/commopdev/docs/en/comm_op_dev_guide/intro.md" target="_blank" rel="noopener noreferrer">Communication Operator Development</a></p>
</td>
</tr>
</tbody>
</table>
</div>

<div class="tablenoborder"><table cellpadding="4" cellspacing="0" summary="" id="EN-US_TOPIC_0000002660762923__table196660351387" frame="border" border="1" rules="all"><caption><b>Table 8 </b>Acceleration libraries</caption><thead align="left"><tr id="EN-US_TOPIC_0000002660762923__row16666235384"><th align="left" class="cellrowborder" valign="top" width="100%" id="mcps1.4.11.2.2.1.1"><p id="EN-US_TOPIC_0000002660762923__p466653517817">Acceleration library</p>
</th>
</tr>
</thead>
<tbody><tr id="EN-US_TOPIC_0000002660762923__row1266613351588"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.11.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p13256747181615"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/acce/SiP/docs/zh/Installation_Operation_Guide/overview.md" target="_blank" rel="noopener noreferrer">Signal Processing Boost (SiP)</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row266623520812"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.11.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p82561547181612"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/acce/dvpp/dvpp_03_0000.html" target="_blank" rel="noopener noreferrer">DVPP Media Acceleration Library</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row96665353820"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.11.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p12256104720164"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/acce/ascendtb/ascendtb_0001.html" target="_blank" rel="noopener noreferrer">Ascend Transformer Boost (ATB)</a></p>
</td>
</tr>
</tbody>
</table>
</div>

<div class="tablenoborder"><table cellpadding="4" cellspacing="0" summary="" id="EN-US_TOPIC_0000002660762923__table19598153619214" frame="border" border="1" rules="all"><caption><b>Table 9 </b>Debugging and analysis tools</caption><thead align="left"><tr id="EN-US_TOPIC_0000002660762923__row95981536824"><th align="left" class="cellrowborder" valign="top" width="100%" id="mcps1.4.12.2.2.1.1"><p id="EN-US_TOPIC_0000002660762923__p65987361020">Tool</p>
</th>
</tr>
</thead>
<tbody><tr id="EN-US_TOPIC_0000002660762923__row2599136628"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.12.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p55991736629"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/optool/MindStudio/26.1.0/en/user_guide/msot_user_guide.md" target="_blank" rel="noopener noreferrer">OpDev Tools</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row15599103612219"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.12.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p5599143611220"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/opcompiletool/atlasop_compiler_07_0001.html" target="_blank" rel="noopener noreferrer">op_compiler</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row9166164373917"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.12.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p15166154323919"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/ModelAccuracyAnalyzer/atlasaccuracy_16_1000.html" target="_blank" rel="noopener noreferrer">msProbe</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row884819408391"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.12.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p784844010392"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0144.html" target="_blank" rel="noopener noreferrer">msProf</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row143057104015"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.12.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p12306712404"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/msMemScope/atlas_msleaks_0001.html" target="_blank" rel="noopener noreferrer">msMemScope</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row321473019405"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.12.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1621463010408"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/migrationtools/atlasfmkt_16_0001.html" target="_blank" rel="noopener noreferrer">msTransplant</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row15599103618217"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.12.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p123846550217"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/atctool/atlasatc_16_0001.html" target="_blank" rel="noopener noreferrer">Ascend Tensor Compiler (ATC)</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row659916365216"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.12.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p559911363218"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/aoe/auxiliarydevtool_aoe_0001.html" target="_blank" rel="noopener noreferrer">Ascend Optimization Engine (AOE)</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row259953620215"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.12.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p45991236123"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/hccltool/HCCLpertest_16_0001.html" target="_blank" rel="noopener noreferrer">HCCL Performance Tester</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row1159913361424"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.12.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p35998361424"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/amct/atlasamct_16_0001.html" target="_blank" rel="noopener noreferrer">Ascend Model Compression Toolkit (AMCT)</a></p>
</td>
</tr>
</tbody>
</table>
</div>

<div class="tablenoborder"><table cellpadding="4" cellspacing="0" summary="" id="EN-US_TOPIC_0000002660762923__table2058120542044" frame="border" border="1" rules="all"><caption><b>Table 10 </b>References</caption><thead align="left"><tr id="EN-US_TOPIC_0000002660762923__row16581154547"><th align="left" class="cellrowborder" valign="top" width="100%" id="mcps1.4.13.2.2.1.1"><p id="EN-US_TOPIC_0000002660762923__p758114541542">Reference</p>
</th>
</tr>
</thead>
<tbody><tr id="EN-US_TOPIC_0000002660762923__row1958175415410"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.13.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p18581754543"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/troubleshooting/troubleshooting_0001.html" target="_blank" rel="noopener noreferrer">Troubleshooting</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row25816542411"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.13.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1158118543415"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/logreference/logreference_0001.html" target="_blank" rel="noopener noreferrer">Log Reference</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row15811541549"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.13.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1158165410412"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/envvar/envref_07_0001.html" target="_blank" rel="noopener noreferrer">Environment Variables</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row75816541148"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.13.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1143520256"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/graphubfusionref/atlasrr_30_0003.html" target="_blank" rel="noopener noreferrer">Graph Fusion and UB Fusion Patterns</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row145811054145"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.13.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p918513264262"><a href="https://hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/aifosl/operatorlist_00097.html" target="_blank" rel="noopener noreferrer">AI Framework Operator List</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row7581205411418"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.13.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p05811254646"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/basicdataapi/atlasopapi_07_00001.html" target="_blank" rel="noopener noreferrer">Basic Data Structures and APIs</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row95813541742"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.13.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p6581205414410"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/refdoc/refer001.html" target="_blank" rel="noopener noreferrer">Version-related Documents</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row1170016361454"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.13.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p15700153618515"><a href="https://www.hiascend.com/document/detail/en/Glossary/gls/gls_0001.html" target="_blank" rel="noopener noreferrer">Glossary</a></p>
</td>
</tr>
</tbody>
</table>
</div>

<div class="tablenoborder"><table cellpadding="4" cellspacing="0" summary="" id="EN-US_TOPIC_0000002660762923__table3979135513512" frame="border" border="1" rules="all"><caption><b>Table 11 </b>More</caption><thead align="left"><tr id="EN-US_TOPIC_0000002660762923__row129791855253"><th align="left" class="cellrowborder" valign="top" width="100%" id="mcps1.4.14.2.2.1.1"><p id="EN-US_TOPIC_0000002660762923__p097905515520">More</p>
</th>
</tr>
</thead>
<tbody><tr id="EN-US_TOPIC_0000002660762923__row9892143810424"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.14.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p14893123824219"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/acldevg/acldevg_0001.html" target="_blank" rel="noopener noreferrer">Application Development</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row169791255453"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.14.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1497912551956"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/fvsearch/aclcppdevg_03_1334.html" target="_blank" rel="noopener noreferrer">Feature Vector Search</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row297915551656"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.14.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p19792557516"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/rpingdev/RPing_02_0001.html" target="_blank" rel="noopener noreferrer">RPing Development</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row197913553511"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.14.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p99792551954"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/tbeaicpudevg/atlasopdev_10_0001.html" target="_blank" rel="noopener noreferrer">TBE &amp; AI CPU Operator Development</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row1897914553510"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.14.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p1397916557519"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/llmdatadistdev/llmdastdev_001.html" target="_blank" rel="noopener noreferrer">LLM DataDist Development</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row0979355355"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.14.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p39792552513"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/dataflowdevg/dataflow_dev_001.html" target="_blank" rel="noopener noreferrer">DataFlow Development</a></p>
</td>
</tr>
<tr id="EN-US_TOPIC_0000002660762923__row89791255551"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.4.14.2.2.1.1 "><p id="EN-US_TOPIC_0000002660762923__p199797551259"><a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/ispdevug/ispdevug_0001.html" target="_blank" rel="noopener noreferrer">ISP Image Tuning</a></p>
</td>
</tr>
</tbody>
</table>
</div>
</div>

</body></html>