# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Deterministic native latency summaries and arithmetic differences."""
from __future__ import annotations

import math
import statistics


def samples(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('measurement_samples_missing')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('measurement_samples_invalid')
    return [float(v) for v in values]


def summary(values):
    values = samples(values)
    median = statistics.median(values)
    mad = statistics.median(abs(v - median) for v in values)
    return {'count': len(values), 'median_us': median, 'min_us': min(values),
        'max_us': max(values), 'mean_us': statistics.fmean(values), 'mad_us': mad,
        'relative_mad': mad / median, 'samples_us': values}


def compare(before, after):
    """Describe observed medians only; no noise model or performance decision."""
    a, b = summary(before), summary(after)
    delta = b['median_us'] - a['median_us']
    return {'before': a, 'after': b, 'delta_us': delta,
            'relative_change': delta / a['median_us'],
            'comparison': 'decreased' if delta < 0 else 'increased' if delta > 0 else 'no_change',
            'interpretation': 'observed_sample_medians', 'official_score': False}
