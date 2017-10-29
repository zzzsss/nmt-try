#

import dynet as dy
import torch as tr
import numpy as np
from .utils import Random
from . import utils
from collections import Iterable

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
    concatenate_to_batch = dy.concatenate_to_batch
    dropout = dy.dropout
    inputTensor = dy.inputTensor
    inputVector = dy.inputVector
    logistic = dy.logistic
    lookup_batch = dy.lookup_batch
    pick_batch_elems = dy.pick_batch_elems
    pick_range = dy.pick_range
    random_bernoulli = dy.random_bernoulli
    reshape = dy.reshape
    softmax = dy.softmax
    tanh = dy.tanh
    transpose = dy.transpose
    zeros = dy.zeros

    @staticmethod
    def new_graph():
        dy.renew_cg()   # new graph

    @staticmethod
    def new_model():
        return dy.ParameterCollection()

    @staticmethod
    def load_model(fname, m):
        m.populate(fname)
        return m

    @staticmethod
    def save_model(fname, m):
        m.save(fname)

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

    # manipulating the batches #
    @staticmethod
    def batch_rearrange(exprs, orders):
        if not isinstance(exprs, Iterable):
            exprs = [exprs]
        if not isinstance(orders, Iterable):
            orders = [orders]
        utils.zcheck_matched_length(exprs, orders, _forced=True)
        new_ones = []
        for e, o in zip(exprs, orders):
            new_ones.append(BK_DY.pick_batch_elems(e, o))
        if len(new_ones) == 1:
            return new_ones[0]
        else:
            return BK_DY.concatenate_to_batch(new_ones)

    @staticmethod
    def batch_repeat(expr, num=1):
        # repeat each element (expanding) in the batch
        utils.zcheck_range(num, 1, None, _forced=True)
        if num == 1:
            return expr
        else:
            bsize = BK_DY.bsize(expr)
            orders = [i//num for i in range(bsize*num)]
            return BK_DY.batch_rearrange(expr, orders)
