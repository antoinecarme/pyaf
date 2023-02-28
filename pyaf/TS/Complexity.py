# Copyright (C) 2023 Antoine Carme <Antoine.Carme@outlook.com>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import Utils as tsutil

from enum import Enum, IntEnum

# strings => avoid strange arithmetics (not additive). Prefer voting or counts/statistics
class eModelComplexity(Enum):
    Low = 'S'
    Medium = 'M'
    High = 'L' # or 'H' ??


