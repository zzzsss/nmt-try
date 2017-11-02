# !! maybe the most important module, the searching/decoding part
# used for both testing and training

from collections import defaultdict, Iterable
import numpy as np
from . import utils

# the searching graph (tracking the relations between states)
class SearchGraph(object):
    def __init__(self):
        self.ch_recs = defaultdict(list)
        self.root = None

    def reg(self, state):
        if state.prev is not None:
            self.ch_recs[state.prev.id].append(state)
        else:
            self.root = state

    def childs(self, state):
        return self.ch_recs[state.id]

    def bfs(self):
        ret = []
        currents = [self.root]
        while len(currents) > 0:
            ret.append(currents)
            nexts = []
            for one in currents:
                nexts += self.childs(one)
            currents = nexts
        return ret

# the states in the searching graph (should be kept small)
class State(object):
    _state_id = 0
    @staticmethod
    def _get_id():
        State._state_id += 1
        return State._state_id
    @staticmethod
    def reset_id():
        State._state_id = 0

    def __init__(self, sg=None, action=None, prev=None, **kwargs):
        self.sg = sg
        self.action = action
        self.prev = prev        # previous state, if None then the start state
        self.length = 0         # length of the actions
        self.score_all = 0      # accumulated scores
        self.ended = False      # whether this state has been ended
        self.values = kwargs    # additional values & information
        self._score_final = None
        self._score_partial = 0
        if prev is not None:
            self.length = prev.length + 1
            self._score_partial = action.score + prev._score_partial
            self.sg = prev.sg
        self.id = State._get_id()   # should be enough within python's int range
        if self.sg is not None:
            self.sg.reg(self)       # register in the search graph

    # for convenience, leave these out
    # @property: action, prev, id, length

    def __repr__(self):
        return "(%s/%d/%.3f)" % (self.action, self.length, self.score_all)

    def __str__(self):
        return self.__repr__()

    @property
    def score_partial(self):
        return self._score_partial

    @property
    def score_final(self):
        return self._score_final

    def set_score_final(self, s):
        utils.zcheck(self.ended, "Nonlegal final calculation for un-end states.")
        self._score_final = s

    @property
    def signature(self):
        return -1

    def mark_end(self):
        self.ended = True

    def is_end(self):
        return self.ended

    # whether this is the starting state
    def is_start(self):
        return self.prev is None

    def get(self, which=None):
        if which is None:
            return self
        elif self.values and which in self.values:
            return self.values[which]
        else:
            return getattr(self, which)

    def get_path(self, which=None):
        if self.prev is None:
            l = []
        else:
            l = self.prev.get_path(which)
            v = self.get(which)
            l.append(v)
        return l

# action
class Action(object):
    def __init__(self, action_code, score, **kwargs):
        self.action_code = action_code
        self.score = score
        self.values = kwargs

    def __repr__(self):
        return "[%s/%.3f/%s]" % (self.action_code, self.score, self.values)

    def __str__(self):
        return self.__repr__()

    def __int__(self):
        return self.action_code

# the scorer who manages the scoring and interacts with the model
# -- be aware of the batch, proper handling to do more batched processing
class Scorer(object):
    def __init__(self):
        State.reset_id()

# results of one step of calculation, including the scores
class Results(object):
    def __init__(self, data, ls, lastd):
        # ls could be int(ls*lastd) or list-of-int(sum(ls)*lastd)
        self._data = data
        self._ls = ls
        self._d = lastd

    def __getitem__(self, item):
        if self._ls is None:
            return self._data[self._start+item]
        else:
            return self._ls[item]

    def __setitem__(self, key, value):
        raise NotImplementedError("Don't need to do this?")

    def argmax(self, n=1):
        utils.zcheck(self._ls is None, "only argmax on smallest one")
        thres = max(-self._d, -n)
        ids = np.argpartition(self._data[self._start: self._start+self._d], thres)[thres:]
        return [int(i) for i in ids]

# state + actions (to be scored, not fully expanded)
class StateCands(object):
    pass

# =========================
# the searcher
class Searcher(object):
    pass

# the loss builder
class Losser(object):
    pass
