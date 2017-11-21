from collections import Iterable
from .common import *
import sys

import _dynet as dy

class DY_CONFIG:
    immediate_compute = False

def init(opts):
    # todo: manipulating sys.argv
    utils.zlog("Using BACKEND of DYNET.")
    params = dy.DynetParams()
    temp = sys.argv
    sys.argv = [temp[0], "--dynet-mem", opts["dynet-mem"], "--dynet-autobatch", opts["dynet-autobatch"],
                "--dynet-devices", opts["dynet-devices"], "--dynet-seed", opts["dynet-seed"]]
    DY_CONFIG.immediate_compute = opts["dynet-immed"]
    params.from_args(None)
    params.init()
    sys.argv = temp

affine = dy.affine_transform
average = concat_wrapper(dy.average)
cmult = dy.cmult
cdiv = dy.cdiv
colwise_add = dy.colwise_add
concatenate = concat_wrapper(dy.concatenate)
concatenate_cols = concat_wrapper(dy.concatenate_cols)
concatenate_to_batch = concat_wrapper(dy.concatenate_to_batch)
dropout = dy.dropout
esum = dy.esum
log = dy.log
inputTensor = dy.inputTensor
inputVector = dy.inputVector
logistic = dy.logistic
lookup_batch = dy.lookup_batch
mean_batches = dy.mean_batches
nobackprop = dy.nobackprop
pickneglogsoftmax_batch = dy.pickneglogsoftmax_batch
pick_batch_elems = dy.pick_batch_elems
pick_batch_elem = dy.pick_batch_elem
pick_range = dy.pick_range
reshape = dy.reshape
softmax = dy.softmax
square = dy.square
sum_batches = dy.sum_batches
tanh = dy.tanh
transpose = dy.transpose
zeros = dy.zeros

def random_bernoulli(rate, size, bsize):
    return dy.random_bernoulli((size,), 1.-rate, 1./(1.-rate), batch_size=bsize)

def new_graph():
    # dy.renew_cg(immediate_compute = True, check_validity = True)   # new graph
    dy.renew_cg(immediate_compute=DY_CONFIG.immediate_compute)

def new_model():
    return dy.ParameterCollection()

def load_model(fname, m):
    m.populate(fname)
    return m

def save_model(fname, m):
    m.save(fname)

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

def param2np(p):
    return p.as_array()

def forward(expr):
    expr.forward()

def backward(expr):
    expr.backward()

def get_value_vec(expr):
    expr.forward()
    return expr.vec_value()

def get_value_sca(expr):
    expr.forward()
    return expr.scalar_value()

def get_value_np(expr):
    expr.forward()
    return expr.npvalue()

def get_params(model, shape, lookup=False, init="default"):
    if isinstance(init, np.ndarray):    # pass it directly
        arr = init
    else:
        arr = get_params_init(shape, init, lookup)
    if lookup:
        p = model.lookup_parameters_from_numpy(arr)
    else:
        p = model.add_parameters(shape, init=dy.NumpyInitializer(arr))
    return p

def gru():
    pass

def vanilla_lstm(iis, hh, cc, px, ph, b, dropx, droph):
    if dropx is not None and droph is not None:
        gates_t = dy.vanilla_lstm_gates_concat_dropout(iis, hh, px, ph, b, dropx, droph)
    else:
        gates_t = dy.vanilla_lstm_gates_concat(iis, hh, px, ph, b)
    cc = dy.vanilla_lstm_c(cc, gates_t)
    hidden = dy.vanilla_lstm_h(cc, gates_t)
    return hidden, cc

def dims(expr):
    return expr.dim()[0]

def bsize(expr):
    return expr.dim()[-1]

# manipulating the batches #

# def batch_rearrange(exprs, orders):
#     if not isinstance(exprs, Iterable):
#         exprs = [exprs]
#     if not isinstance(orders, Iterable):
#         orders = [orders]
#     utils.zcheck_matched_length(exprs, orders, _forced=True)
#     new_ones = []
#     for e, o in zip(exprs, orders):
#         new_ones.append(pick_batch_elems(e, o))
#     return concatenate_to_batch(new_ones)

def batch_rearrange_one(e, o):
    return pick_batch_elems(e, o)

def batch_repeat(expr, num=1):
    # repeat each element (expanding) in the batch
    utils.zcheck_range(num, 1, None, _forced=True)
    if num == 1:
        return expr
    else:
        bs = bsize(expr)
        orders = [i//num for i in range(bs*num)]
        return batch_rearrange_one(expr, orders)

class Trainer(object):
    def __init__(self, model, type, lrate, moment=None):
        self._tt = {"sgd": dy.SimpleSGDTrainer(model, lrate),
                        "momentum": dy.MomentumSGDTrainer(model, lrate, moment),
                        "adam": dy.AdamTrainer(model, lrate)
                        }[type]

    def restart(self):
        self._tt.restart()

    def set_lrate(self, lr):
        self._tt.learning_rate = lr

    def set_clip(self, cl):
        self._tt.set_clip_threshold(cl)

    def save_shadow(self, fname):
        # todo
        utils.zcheck(False, "Not implemented for saving shadows.", func="warn")

    def load_shadow(self, fname):
        # todo
        utils.zcheck(False, "Not implemented for loading shadows.", func="warn")

    def update(self):
        self._tt.update()

def rearrange_cache(cache, order):
    if isinstance(cache, dict):
        ret = {}
        for n in cache:
            ret[n] = rearrange_cache(cache[n], order)
        return ret
    elif isinstance(cache, list):
        return [rearrange_cache(_i, order) for _i in cache]
    elif isinstance(cache, type(None)):
        return None
    else:
        return batch_rearrange_one(cache, order)

def recombine_cache(caches, indexes):
    # input lists, output combine one. todo: to be more efficient
    c0 = caches[0]
    if isinstance(c0, dict):
        ret = {}
        for n in c0:
            ret[n] = recombine_cache([_c[n] for _c in caches], indexes)
        return ret
    elif isinstance(c0, list):
        return [recombine_cache([_c[_i] for _c in caches], indexes) for _i in range(len(c0))]
    elif isinstance(c0, type(None)):
        return None
    else:
        them = [pick_batch_elem(_c, _i) for _c, _i in zip(caches, indexes)]
        return concatenate_to_batch(them)
