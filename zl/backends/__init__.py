import os
from .common import COMMON_CONFIG, utils

# about backend
# Set backend based on KERAS_BACKEND flag, if applicable.
if 'ZL_BACKEND' in os.environ:
    _backend = os.environ['ZL_BACKEND']
    _BACKEND = {'dy':'dy', 'dynet':'dy', 'tr':'tr', 'torch':'tr', 'pytorch':'tr'}[_backend]
else:
    # Default backend: dynet.
    _BACKEND = 'dy'

## import them all
if _BACKEND == 'tr':
    from . import bktr as BK
elif _BACKEND == 'dy':
    from . import bkdy as BK
else:
    pass

def init_bk(opts):
    to_set = {}
    for i in COMMON_CONFIG.values:
        if i in opts:
            to_set[i] = opts[i]
    for k in to_set:
        COMMON_CONFIG.values[k] = to_set[k]
    utils.zlog("For inits: %s" % COMMON_CONFIG.values)
    BK.init(opts)
