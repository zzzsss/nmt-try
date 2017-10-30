from collections import Iterable
from .common import *
import sys

import _dynet as dy
def init(opts):
    # todo: manipulating sys.argv
    utils.zlog("Using BACKEND of DYNET.")
    params = dy.DynetParams()
    temp = sys.argv
    sys.argv = ["--dynet-mem", opts["dynet-mem"], "--dynet-autobatch", opts["dynet-autobatch"],
                "--dynet-devices", opts["dynet-devices"], "--dynet-seed", opts["dynet-seed"]]
    params.from_args(None)
    params.init()
    sys.argv = temp

affine = dy.affine_transform
average = concat_wrapper(dy.average)
cmult = dy.cmult
colwise_add = dy.colwise_add
concatenate = concat_wrapper(dy.concatenate)
concatenate_cols = concat_wrapper(dy.concatenate_cols)
concatenate_to_batch = concat_wrapper(dy.concatenate_to_batch)
dropout = dy.dropout
esum = dy.esum
inputTensor = dy.inputTensor
inputVector = dy.inputVector
logistic = dy.logistic
lookup_batch = dy.lookup_batch
pickneglogsoftmax_batch = dy.pickneglogsoftmax_batch
pick_batch_elems = dy.pick_batch_elems
pick_range = dy.pick_range
random_bernoulli = dy.random_bernoulli
reshape = dy.reshape
softmax = dy.softmax
sum_batches = dy.sum_batches
tanh = dy.tanh
transpose = dy.transpose
zeros = dy.zeros

def new_graph():
    dy.renew_cg()   # new graph

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

def get_params(model, shape, lookup=False, init="default"):
    if isinstance(init, np.ndarray):    # pass it directly
        arr = init
    else:
        arr = get_params_init(shape, init)
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

def batch_rearrange(exprs, orders):
    if not isinstance(exprs, Iterable):
        exprs = [exprs]
    if not isinstance(orders, Iterable):
        orders = [orders]
    utils.zcheck_matched_length(exprs, orders, _forced=True)
    new_ones = []
    for e, o in zip(exprs, orders):
        new_ones.append(pick_batch_elems(e, o))
    return concatenate_to_batch(new_ones)

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
        return batch_rearrange(expr, orders)

class Trainer(object):
    def __init__(self, model, type, lrate, moment=None):
        self._tt = {"sgd": dy.SimpleSGDTrainer(model, lrate),
                        "momentum": dy.MomentumSGDTrainer(model, lrate, moment),
                        "adam": dy.AdamTrainer(model, lrate)
                        }[type]

    def restart(self):
        self._tt.restrat()

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
