#     #####             ####   ######       PyAF
#     ##  ##  ##   ##  ##  ##  ##           Python Automatic Forecasting
#     #####    ## ##   ######  ####   
#     ##        ##     ##  ##  ##           Version 5.x
#     ##       ##      ##  ##  ##           https://github.com/antoinecarme/pyaf
#             ##
# SPDX-FileCopyrightText: Copyright (c) (2017-) Antoine CARME <Antoine.Carme@outlook.com>
# SPDX-License-Identifier: BSD-3-Clause ( https://spdx.org/licenses/BSD-3-Clause.html )


def check_python_version_for_pyaf():
    import six
    if six.PY2:
        raise Exception("PYAF_ERROR_PYTHON_2_NOT_SUPPORTED")


check_python_version_for_pyaf()

from . import ForecastEngine, HierarchicalForecastEngine

__version__ = '5.0'

def activate_timer_logging():
    import pyaf.TS.Utils as tsutil
    tsutil.activate_timer_logging()

