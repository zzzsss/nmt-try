# the general components of a neural model
# -- similar to Keras, but with dynamic toolkit as backends

from . import utils
from collections import Iterable, Sized
from .backends import BK_DY as BK

# ================= Basic Blocks ================= #
# basic unit (stateful about dropouts)
class Layer(object):
    def __init__(self, model):
        # basic ones: mainly the parameters
        self.model = model
        self.params = {}
        self.iparams = {}
        self.update = None
        # aux info like dropouts/masks (could be refreshed)
        self.odrop = 0.     # output drops
        self.idrop = 0.     # input drops
        self.gdrop = 0.     # special recurrent drops
        self.gmasks = None  # special masks for gdrop (pre-defined drops)
        self.bsize = None   # bsize for one f/b

    def _refresh(self, argv):
        # to be overridden
        pass

    def refresh(self, **argv):
        # update means whether the parameters should be updated
        update = bool(argv["update"]) if "update" in argv else True
        ingraph = bool(argv["ingraph"]) if "ingraph" in argv else True
        if ingraph:
            for k in self.params:
                self.iparams[k] = BK.param2expr(self.params[k], update)
            self.update = update
        # dropouts
        self.odrop = float(argv["odrop"]) if "odrop" in argv else 0.
        self.idrop = float(argv["idrop"]) if "idrop" in argv else 0.
        self.gdrop = float(argv["gdrop"]) if "gdrop" in argv else 0.
        self.gmasks = None
        self.bsize = int(argv["bsize"]) if "bsize" in argv else None
        # maybe others
        self._refresh(argv)

    def _add_params(self, shape, lookup=False, init="default"):
        return BK.get_params(self.model, shape, lookup, init)

# linear layer with selectable activation functions
class Affine(Layer):
    _ACTS = ["linear", "tanh", "softmax"]
    _ACT_DEFAULT = "linear"

    def __init__(self, model, n_ins, n_out, act="tanh", bias=True):
        super(Affine, self).__init__(model)
        # list of n_ins and n_outs have different meanings: horizontal and vertical
        if not isinstance(n_ins, Iterable):
            n_ins = [n_ins]
        # dimensions
        self.n_ins = n_ins
        self.n_out = n_out
        # activations
        self.act = act
        self._act_ffs = BK.getf(self.act)
        # params
        self.bias = bias
        for i, din in enumerate(n_ins):
            self.params["W"+str(i)] = self._add_params((n_out, din))
        if bias:
            self.params["B"] = self._add_params((n_out,))

    def __repr__(self):
        return "# Affine (%s -> %s [%s])" % (self.n_ins, self.n_out, self.act)

    def __str__(self):
        return self.__repr__()

    def __call__(self, input_exp):
        if not isinstance(input_exp, Iterable):
            input_exp = [input_exp]
        if self.bias:
            input_lists = [self.iparams["B"]]
        else:
            input_lists = [BK.zeros(self.n_out)]
        for i, one_inp in enumerate(input_exp):
            input_lists += [self.iparams["W"+str(i)], one_inp]
        h0 = BK.affine(input_lists)
        h1 = self._act_ffs(h0)
        if self.odrop > 0.:
            h1 = BK.dropout(x, self.odrop)
        return h1
