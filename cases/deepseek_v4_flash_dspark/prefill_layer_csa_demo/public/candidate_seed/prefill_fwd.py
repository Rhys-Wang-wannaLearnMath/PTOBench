# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run
# ci: no-sim    # CI marker: full multi-layer / multi-card forward — device-only, skip on *sim

"""REF closure for deepseek_v4_flash_dspark.prefill_layer_csa_demo; fixed bindings from public config."""
import pypto.language as pl

FWD_TOKENS_DYN = pl.dynamic("PREFILL_FWD_TOKENS_DYN")
FWD_GROUP_TOKENS_DYN = pl.dynamic("PREFILL_FWD_GROUP_TOKENS_DYN")

