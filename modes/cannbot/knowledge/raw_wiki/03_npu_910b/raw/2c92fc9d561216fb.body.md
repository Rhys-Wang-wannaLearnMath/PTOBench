<!doctype html>
<html>
 <head>
  <meta charset="utf-8">
  <meta name="search_title" content="">
 </head>
 <body>
  <a name="ZH-CN_TOPIC_0000002562272921"></a><a name="ZH-CN_TOPIC_0000002562272921"></a>
  <h1 class="topictitle1">什么是Ascend C</h1>
  <div id="body8662426">
   <p id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_p8748853101610">Ascend C是CANN针对算子开发场景推出的编程语言，原生支持C和C++标准规范，兼具开发效率和运行性能。基于Ascend C编写的算子程序，通过编译器编译和运行时调度，运行在昇腾AI处理器上。使用Ascend C，开发者可以基于昇腾AI硬件，高效的实现自定义的创新算法。您可以通过<a href="https://www.hiascend.com/zh/ascend-c" target="_blank" rel="noopener noreferrer">Ascend C主页</a>了解更详细的内容。</p>
   <p id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_p121541701352">Ascend C提供多层级API，满足多维场景算子开发诉求。</p>
   <ul id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_ul82095603618">
    <li id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_li31921291134"><strong id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_b02010568362">语言扩展层 C API</strong>：开放芯片完备编程能力，支持数组分配内存，一般基于指针编程，提供与业界一致的C语言编程体验。</li>
    <li id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_li72025683611"><strong id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_b52055633620">基础API</strong>：基于Tensor进行编程的C++类库API，实现单指令级抽象，为底层算子开发提供灵活控制能力。</li>
    <li id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_li1147537373"><strong id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_b020956143611">高阶API：</strong>封装单核公共算法，涵盖一些常见的计算算法（如卷积、矩阵运算等），显著降低复杂算法开发门槛。</li>
    <li id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_li1620155613613"><strong id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_b1720195611368">算子模板库：</strong>基于模板提供算子完整实现参考，简化Tiling（切分算法）开发，支撑用户自定义扩展。</li>
    <li id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_li32015663619"><strong id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_b3201056193612">Python前端：</strong><a href="https://gitcode.com/cann/pyasc" target="_blank" rel="noopener noreferrer">PyAsc编程语言</a>基于Python原生接口，提供芯片底层完备编程能力，支持基于Python接口开发高性能Ascend C算子。</li>
   </ul>
   <p id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_p29495713715"></p>
   <p id="ZH-CN_TOPIC_0000002562272921__p121451353154216"><span><img class="eddx" id="ZH-CN_TOPIC_0000002562272921__image11729175394210" src="figure/zh-cn_image_0000002531512958.png"></span></p>
   <div class="section" id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_section11489186191713">
    <h4 class="sectiontitle">快速入门</h4>
    <div id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_imagemap358012510268">
     <img usemap="#d86049e66" border="0" src="figure/zh-cn_image_0000002562432945.png"> <map name="d86049e66" id="d86049e66"> <area href="atlas_ascendc_map_10_0016.html" alt="" title="" shape="rect" coords="722,19,1032,72"> </map>
    </div>
    <p id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_p122031047121113"></p>
   </div>
   <div class="section" id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_section181811514171911">
    <h4 class="sectiontitle">成长地图</h4>
    <div id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_imagemap1963611449476">
     <img usemap="#d86049e79" border="0" src="figure/zh-cn_image_0000002562432937.png"> <map name="d86049e79" id="d86049e79"> <area href="atlas_ascendc_map_10_0003.html" alt="" title="" shape="rect" coords="168,49,304,125"> <area href="atlas_ascendc_10_10063.html" alt="" title="" shape="rect" coords="166,209,309,246"> <area href="atlas_ascendc_10_10051.html" alt="" title="" shape="rect" coords="314,201,410,249"> <area href="atlas_ascendc_10_0032.html" alt="" title="" shape="rect" coords="195,349,278,399"> <area href="atlas_ascendc_10_0036.html" alt="" title="" shape="rect" coords="298,368,387,400"> <area href="atlas_ascendc_10_0042.html" alt="" title="" shape="rect" coords="403,361,492,399"> <area href="atlas_ascendc_10_0048.html" alt="" title="" shape="rect" coords="516,346,597,399"> <area href="atlas_ascendc_10_00036.html" alt="" title="" shape="rect" coords="634,337,712,398"> <area href="atlas_ascendc_10_00043.html" alt="" title="" shape="rect" coords="787,304,879,374"> <area href="atlas_ascendc_10_0073.html" alt="" title="" shape="rect" coords="173,509,267,566"> <area href="atlas_ascendc_10_0075.html#ZH-CN_TOPIC_0000002531522168__section166162435313" alt="通过printf打印数据" title="通过printf打印数据" shape="rect" coords="301,538,427,574"> <area href="atlas_ascendc_10_00059.html" alt="" title="" shape="rect" coords="775,462,949,544"> <area href="atlas_ascendc_10_00043.html" alt="" title="" shape="rect" coords="188,627,426,718"> <area href="atlas_ascendc_10_0057.html" alt="" title="" shape="rect" coords="708,661,898,720"> </map>
    </div>
    <p id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_p9999204118273"></p>
   </div>
   <div class="section" id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_section126809373191">
    <h4 class="sectiontitle">概念原理</h4>
    <div id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_imagemap8680143711195">
     <img usemap="#d86049e183" border="0" src="figure/zh-cn_image_0000002562272941.png"> <map name="d86049e183" id="d86049e183"> <area href="atlas_ascendc_10_0097.html" alt="" title="" shape="rect" coords="35,80,237,142"> <area href="atlas_ascendc_10_00015.html" alt="" title="" shape="rect" coords="272,83,496,144"> <area href="atlas_ascendc_10_00030.html" alt="" title="" shape="rect" coords="541,88,726,140"> <area href="atlas_ascendc_10_00016.html" alt="" title="" shape="rect" coords="762,81,957,146"> </map>
    </div>
    <p id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_p1968019379197"></p>
   </div>
   <div class="section" id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_section116808372193">
    <h4 class="sectiontitle">API参考</h4>
    <div id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_imagemap176801737151915">
     <img usemap="#d86049e217" border="0" src="figure/zh-cn_image_0000002562272929.png"> <map name="d86049e217" id="d86049e217"> <area href="/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0003.html#ZH-CN_TOPIC_0000002531382050__section117632211201" alt="基础API" title="基础API" shape="rect" coords="46,85,222,144"> <area href="/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0003.html#ZH-CN_TOPIC_0000002531382050__section3317105813235" alt="高阶API" title="高阶API" shape="rect" coords="242,78,442,143"> <area href="/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0003.html#ZH-CN_TOPIC_0000002531382050__section15221943104512" alt="Utils API" title="Utils API" shape="rect" coords="677,80,880,147"> <area href="/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0426.html" alt="" title="" shape="rect" coords="473,74,664,146"> </map>
    </div>
    <p id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_p5680153761914"></p>
    <p id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_p219346102411"></p>
   </div>
   <div class="section" id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_section16680133716195">
    <h4 class="sectiontitle">算子实践参考</h4>
    <div id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_imagemap46802037161920">
     <img usemap="#d86049e252" border="0" src="figure/zh-cn_image_0000002562432943.png"> <map name="d86049e252" id="d86049e252"> <area href="atlas_ascendc_best_practices_10_0010.html" alt="" title="" shape="rect" coords="52,85,221,218"> <area href="atlas_ascendc_best_practices_10_0040.html" alt="" title="" shape="rect" coords="272,98,478,209"> <area href="atlas_ascendc_best_practices_10_10002.html" alt="" title="" shape="rect" coords="531,103,718,213"> <area href="atlas_ascendc_best_practices_10_0043.html" alt="" title="" shape="rect" coords="767,99,936,201"> <area href="atlas_ascendc_best_practices_10_0001.html" alt="" title="" shape="rect" coords="881,21,962,58"> </map>
    </div>
    <p id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_p368053715197"></p>
   </div>
   <p id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_p16549236193110">Ascend C支持在如下AI处理器型号使用：</p>
   <ul id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_ul723415519312">
    <li id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_li10800162235118"><span id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_ph2272194216543">Atlas 350 加速卡</span></li>
    <li id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_li17391143711246"><span id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_ph16239174011416"><term id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002562275259_term1253731311225">
       Atlas A3 训练系列产品
      </term>/<term id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002562275259_term131434243115">
       Atlas A3 推理系列产品
      </term></span></li>
    <li id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_li519213411807"><span id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_ph1292674871116"><term id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002562275259_term11962195213215">
       Atlas A2 训练系列产品
      </term>/<term id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002562275259_term184716139811">
       Atlas A2 推理系列产品
      </term></span></li>
   </ul>
   <ul id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_ul4445227328">
    <li id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_li033144914353"><span id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_ph15301744513"><term id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002562275259_term354143892110">
       Atlas 200I/500 A2 推理产品
      </term></span></li>
    <li id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_li196597131911"><span id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_ph19951183013136"><term id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002562275259_term4363218112215">
       Atlas 推理系列产品
      </term></span></li>
    <li id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_li1316571514912"><span id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002484241782_ph9594183016437"><term id="ZH-CN_TOPIC_0000002562272921__zh-cn_topic_0000002562275259_term71949488213">
       Atlas 训练系列产品
      </term></span></li>
   </ul>
  </div>
  <div>
   <div class="familylinks">
    <div class="parentlink">
     <strong>父主题：</strong> <a href="atlas_ascendc_map_10_0001.html">入门教程</a>
    </div>
   </div>
  </div>
 </body>
</html>