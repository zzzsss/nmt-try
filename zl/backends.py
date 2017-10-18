#

import dynet as dy
import torch as tr
import numpy as np
from .utils import Random

# first init params with np: return is C-order as the default of numpy
# -- also fix some hyper-params here
def _get_params_init(shape, init):
    # shape is a tuple of dims
    assert init in ["default", "random", "glorot", "ortho", "gaussian"], "Unknown init method %s" % init
    if len(shape) == 1:     # set bias to 0
        return np.array([0. for _ in range(shape[0])])
    elif len(shape) == 2:
        if init == "default" or init == "glorot":
            w0 = Random.rand(shape, "weights")  # [0,1)
            w0 = (w0-0.5)*2*(np.sqrt(6.0/(sum(shape))))
            return w0
        elif init == "random":
            w0 = Random.rand(shape, "weights")  # [0,1)
            w0 = (w0-0.5)*2*0.01    # scale = 0.01
            return w0
        elif init == "gaussian":
            w0 = Random.randn(shape, "weights")
            w0 *= 0.01  # var = 0.01^2
            return w0
        elif init == "ortho":
            assert shape[0]%shape[1] == 0, "Bad shape %s for ortho_init" % shape
            num = shape[0] // shape[1]
            w0 = Random.ortho_weight(shape[1], "weights") if num == 1 else\
                  np.concatenate([Random.ortho_weight(shape[1], "weights") for _ in range(num)])
            return w0
    else:
        raise NotImplementedError("Currently only support parameter dim <= 2.")

class BK_DY:
    @staticmethod
    def init():
        pass

    affine = dy.affine_transform
    average = dy.average
    cmult = dy.cmult
    colwise_add = dy.colwise_add
    concatenate = dy.concatenate
    concatenate_cols = dy.concatenate_cols
    dropout = dy.dropout
    inputTensor = dy.inputTensor
    inputVector = dy.inputVector
    logistic = dy.logistic
    lookup_batch = dy.lookup_batch
    random_bernoulli = dy.random_bernoulli
    reshape = dy.reshape
    softmax = dy.softmax
    tanh = dy.tanh
    transpose = dy.transpose
    zeros = dy.zeros

    @staticmethod
    def param2expr(p, update):
        # todo(warn): dynet changes API
        try:
            e = dy.parameter(p, update)
        except NotImplementedError:
            if update:
                e = dy.parameter(p)
            else:
                e = dy.const_parameter(p)
        return e

    @staticmethod
    def get_params(model, shape, lookup=False, init="default"):
        if isinstance(init, np.ndarray):    # pass it directly
            arr = init
        else:
            arr = _get_params_init(shape, init)
        if lookup:
            p = model.lookup_parameters_from_numpy(arr)
        else:
            p = model.add_parameters(shape, init=dy.NumpyInitializer(arr))
        return p

    @staticmethod
    def gru():
        pass

    @staticmethod
    def vanilla_lstm(iis, hh, cc, px, ph, b, dropx, droph):
        if dropx is not None and droph is not None:
            gates_t = dy.vanilla_lstm_gates_concat_dropout(iis, hh, px, ph, b, dropx, droph)
        else:
            gates_t = dy.vanilla_lstm_gates_concat(iis, hh, px, ph, b)
        cc = dy.vanilla_lstm_c(cc, gates_t)
        hidden = dy.vanilla_lstm_h(cc, gates_t)
        return hidden, cc

    @staticmethod
    def dims(expr):
        return expr.dim()[0]

    @staticmethod
    def bsize(expr):
        return expr.dim()[-1]