# the model which contains the params and could be used to score candidates

from . import layers, utils
from collections import Iterable

class Model(object):
    def __init__(self):
        utils.zlog("Start to create Model.")
        # init models
        self.model = layers.BK.new_model()
        # self.nodes = []
        # general computation process: according to the values in the caches
        self.names_bv = set()       # beam variant (need reshuffle within batches)
        self.names_bi = set()       # beam invariant (only controlling the size for each instance will be fine)
        self.names_ig = set()       # ignored (not used in the next steps)

    @staticmethod
    def new_graph():
        layers.BK.new_graph()

    def refresh(self, training):
        # should be called after a new graph and before building the graph
        # default: ingraph=True, update=True
        # def _gd(drop):  # get dropout
        #     return drop if training else 0.
        # # disable drop when training; todo(warn) specific names
        # for k in argv:
        #     if k[1:].startwith("drop"):
        #         argv[k] = _gd(argv[k])
        # for n in self.nodes:
        #     n.refresh(argv)
        raise NotImplementedError("Should specify this in specific models!")

    # save and load #
    def load(self, fname):
        self.model = layers.BK.load_model(fname, self.model)
        utils.zlog("Read Model from %s." % fname, func="io")

    def save(self, fname):
        layers.BK.save_model(fname, self.model)
        utils.zlog("Save Model to %s." % fname, func="io")

    # main routines #
    def start(self, xs, repeat_time):
        raise NotImplementedError("Should specify this in specific models!")

    def step(self, prev_val, inputs):
        raise NotImplementedError("Should specify this in specific models!")
