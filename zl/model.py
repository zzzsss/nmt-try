# the model which contains the params and could be used to score candidates

from . import layers, utils

class Model(object):
    def __init__(self):
        utils.zlog("Start to create Model.")
        # init models
        self.model = layers.BK.new_model()
        self.nodes = []

    @staticmethod
    def new_graph():
        layers.BK.new_graph()

    def refresh(self, training, **argv):
        # should be called after a new graph and before building the graph
        # default: ingraph=True, update=True
        def _gd(drop):  # get dropout
            return drop if training else 0.
        # disable drop when training; todo(warn) specific names
        for k in argv:
            if k[1:].startwith("drop"):
                argv[k] = _gd(argv[k])
        for n in self.nodes:
            n.refresh(argv)

    # save and load #
    def load(self, fname):
        self.model = layers.BK.load_model(fname, self.model)
        utils.zlog("Read Model from %s." % fname, func="io")

    def save(self, fname):
        layers.BK.save_model(fname, self.model)
        utils.zlog("Save Model to %s." % fname, func="io")

    # calculate
    def calculate(self, **argv):
        raise NotImplementedError("No calculate method for basic Model.")
