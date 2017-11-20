# !! maybe the most important module, the searching/decoding part
# used for both testing and training

from collections import defaultdict, Iterable
import numpy as np
from . import utils, data

# the searching graph (tracking the relations between states)
class SearchGraph(object):
    def __init__(self):
        self.ch_recs = defaultdict(list)
        self.root = None
        self.ends = []

    def reg(self, state):
        if state.prev is not None:
            self.ch_recs[state.prev.id].append(state)
        else:
            self.root = state

    def childs(self, state):
        return self.ch_recs[state.id]

    def add_end(self, state):
        self.ends.append(state)

    def get_ends(self):
        return self.ends

    def is_pruned(self, one):
        return not one.is_end() and len(self.childs(one)) == 0

    def bfs(self):
        ret = []        # stayed or ended, pruned
        currents = [self.root]
        while len(currents) > 0:
            nexts = []
            ret.append(([], []))
            for one in currents:
                expands = self.childs(one)
                nexts += expands
                if not self.is_pruned(one):
                    ret[-1][0].append(one)
                else:
                    ret[-1][1].append(one)
                # sorting
                ret[-1][0].sort(key=lambda x: x.score_partial, reverse=True)
                ret[-1][1].sort(key=lambda x: x.score_partial, reverse=True)
            currents = nexts
        return ret

    def show_graph(self, td):
        s = "\n"
        currents = [self.root]
        while len(currents) > 0:
            nexts = []
            for one in currents:
                head = "id=%s|pid=%s|s=%.3f|(%s)" % (one.id, one.pid, one.score_partial, " ".join(data.Vocab.i2w(td, one.get_path("action_code"))))
                expands = self.childs(one)
                exp_strs = []
                for z in sorted(expands, key=lambda x: x.score_partial, reverse=True):
                    if self.is_pruned(z):
                        pr_str = "PR"
                    else:
                        pr_str = "ST"
                        nexts.append(z)
                    exp_strs.append("id=%s|w=%s|s=%.3f(%.3f)|%s" % (z.id, td.getw(z.action_code), z.action.score, np.exp(z.action.score), pr_str))
                head2 = "\n-> " + ", ".join(exp_strs) + "\n"
                s += head + head2
            s += "\n"
            currents = nexts
        utils.zlog(s)

    def analyze(self):
        # return statistics
        stat = {}
        return stat

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
        return "ID=%s|PID=%s|LEN=%s|SS=%.3f(%.3f)|ACT=%s" % (self.id, self.pid, self.length, self._score_partial, self.score_final, self.action)

    def __str__(self):
        return self.__repr__()

    @property
    def pid(self):
        if self.prev is None:
            return -1
        else:
            return self.prev.id

    @property
    def score_partial(self):
        return self._score_partial

    @property
    def score_final(self):
        return utils.Constants.MIN_V if self._score_final is None else self._score_final

    # for the actions
    @property
    def action_code(self):
        return int(self.action)

    def action_score(self, s=None):
        if s is not None:
            self.action.score = s
        return self.action.score

    def set_score_final(self, s):
        utils.zcheck(self.ended, "Nonlegal final calculation for un-end states.")
        self._score_final = s

    def mark_end(self):
        self.ended = True
        self.sg.add_end(self)

    def is_end(self):
        return self.ended

    # whether this is the starting state
    def is_start(self):
        return self.prev is None

    # about values
    def set(self, k, v):
        self.values[k] = v

    def add_list(self, k, v):
        if k not in self.values:
            self.values[k] = []
        utils.zcheck_type(self.values[k], list)
        self.values[k].append(v)

    def transfer_values(self, other, ruler=lambda x: str.islower(x[0])):
        # todo(warn): specific rules, default start with lowercase
        r = {}
        for k in self.values:
            if ruler(k):
                r[k] = self.values[k]
        other.values = r

    def get(self, which=None):
        if which is None:
            return self
        elif self.values and which in self.values:
            return self.values[which]
        elif hasattr(self, which):
            return getattr(self, which)
        else:
            return None

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
    def __init__(self, action_code, score,):
        self.action_code = action_code
        self.score = score
        # self.values = kwargs

    def __repr__(self):
        return "[%s/%.3f/%.3f]" % (self.action_code, self.score, np.exp(self.score))

    def __str__(self):
        return self.__repr__()

    def __int__(self):
        return self.action_code

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
