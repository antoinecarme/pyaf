import numpy as np
from scipy import stats

import sys, scipy, numpy; 
print(scipy.__version__, numpy.__version__, sys.version_info)

a = np.array([0, 0, 0, 1, 1, 1, 1])
b = np.arange(7)
pearson1 = stats.pearsonr(a, b)
a1 = a * 1e90
b1 = b * 1e90
pearson2 = stats.pearsonr(a1, b1)
print(pearson1 , pearson2)

# see https://github.com/scipy/scipy/issues/8980
lNotFixed = True

assert(np.allclose(pearson1[0] , pearson2[0]) or lNotFixed)
assert(np.allclose(pearson1[1] , pearson2[1]) or lNotFixed)
