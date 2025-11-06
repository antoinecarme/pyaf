#     #####             ####   ######       PyAF
#     ##  ##  ##   ##  ##  ##  ##           Python Automatic Forecasting
#     #####    ## ##   ######  ####   
#     ##        ##     ##  ##  ##           Version 5.x
#     ##       ##      ##  ##  ##           https://github.com/antoinecarme/pyaf
#             ##
# SPDX-FileCopyrightText: Copyright (c) (2017-) Antoine CARME <Antoine.Carme@outlook.com>
# SPDX-License-Identifier: BSD-3-Clause ( https://spdx.org/licenses/BSD-3-Clause.html )


import pandas as pd
import numpy as np

from . import Utils as tsutil

from enum import Enum, IntEnum

# strings => avoid strange arithmetics (not additive). Prefer voting or counts/statistics
class eModelComplexity(Enum):
    Low = 'S'
    Medium = 'M'
    High = 'L' # or 'H' ??


