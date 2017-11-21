import numpy as np
from .. import utils
from ..utils import Random

class COMMON_CONFIG(object):
    # todo(warn): scale only useful for random/gaussian
    enabled = False
    values = {
        "bk_init_nl": "glorot", "bk_init_l": "glorot",
        "bk_init_scale_nl": 0.01, "bk_init_scale_l": 0.01
    }

# first init params with np: return is C-order as the default of numpy
# -- also fix some hyper-params here
def get_params_init(shape, init, lookup):
    assert COMMON_CONFIG.enabled    # todo(warn) using this kind of init
    # shape is a tuple of dims
    assert init in ["default", "random", "glorot", "ortho", "gaussian"], "Unknown init method %s" % init
    poss_scale = COMMON_CONFIG.values["bk_init_scale_l"] if lookup else COMMON_CONFIG.values["bk_init_scale_nl"]
    if len(shape) == 1:     # set bias to 0
        return np.array([0. for _ in range(shape[0])])
    elif len(shape) == 2:
        # get defaults
        if init == "default":
            init = COMMON_CONFIG.values["bk_init_l"] if lookup else COMMON_CONFIG.values["bk_init_nl"]
        # specifics
        if init == "glorot":
            if lookup:  # special for lookups
                shape_g = (shape[1], )  # fan-out for lookup
            else:
                shape_g = shape
            w0 = Random.rand(shape, "weights")  # [0,1)
            w0 = (w0-0.5)*2*(np.sqrt(3.0*len(shape_g)/(sum(shape_g))))
            return w0
        elif init == "random":
            w0 = Random.rand(shape, "weights")  # [0,1)
            w0 = (w0-0.5)*2*poss_scale
            return w0
        elif init == "gaussian":
            w0 = Random.randn_clip(shape, "weights")
            w0 *= poss_scale  # var = scale^2
            return w0
        elif init == "ortho":
            assert shape[0]%shape[1] == 0, "Bad shape %s for ortho_init" % shape
            num = shape[0] // shape[1]
            if num == 1:
                w0 = Random.ortho_weight(shape[1], "weights")
            else:
                w0 = np.concatenate([Random.ortho_weight(shape[1], "weights") for _ in range(num)])
            return w0
    raise NotImplementedError("Currently only support parameter dim <= 2.")

def concat_wrapper(ff):
    # ff(xs, *args)
    def _ff(xs, *args):
        if len(xs) == 1:
            return xs[0]
        else:
            return ff(xs, *args)
    return _ff
