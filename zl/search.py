# !! maybe the most important module, the searching/decoding part
# used for both testing and training

from collections import defaultdict

# the searching graph (tracking the relations between states)
class SearchGraph(object):
    def __init__(self):
        self.num = 0
        self.ch_recs = defaultdict(list)

    def reg(self, state):
        self.num += 1
        state.id = self.num
        if state.prev is not None:
            self.ch_recs[state.prev.id].append(state)

# the states in the searching graph (should be kept small)
class State(object):
    def __init__(self, sg, action=None, prev=None, values=None):
        self.action = action
        self.prev = prev        # previous state, if None then the start state
        self.values = values    # additional values
        self.length = 0         # length of the actions
        self.score_all = 0      # accumulated scores
        if prev is not None:
            self.length += prev.length
            self.score_all = action.score + prev.score_all
        self.id = -1
        sg.reg(self)            # register in the search graph

    @property
    def score(self):
        return self.score_all

    @property
    def signature(self):
        return -1

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
    def __init__(self, score):
        self.score = score

    def score(self):
        return self.score

# the scorer who manages the scoring and interacts with the model
# -- be aware of the batch, proper handling to do more batched processing
class Scorer(object):
    pass

# results of one step of calculation, including the scores
class Results(object):
    pass

# state + actions (to be scored, not fully expanded)
class StateCandidates(object):
    pass

# the searcher
class Searcher(object):
    pass
