# !! maybe the most important module, the searching/decoding part
# used for both testing and training

# the states in the searching graph (should be kept small)
class State(object):
    def __init__(self, action, prev=None, values=None):
        self.action = action
        self.prev = prev        # previous state, None as the start state
        self.values = values    # additional values
        self.length = 1         # length of the actions
        self.score_all = action.score()     # accumulated scores
        if prev is not None:
            self.length += prev.length
            self.score_all += prev.score_all

    @property
    def score(self):
        return self.score_all

    @property
    def signature(self):
        return 0

    def get(self, which=None):
        if which is None:
            return self
        elif self.values and which in self.values:
            return self.values[which]
        else:
            return getattr(self, which)

    def get_path(self, which=None):
        v = self.get(which)
        if self.prev is None:
            l = [v]
        else:
            l = self.prev.get_path(which)
            l.append(v)
        return l

# light-weighted state or state+action (not fully expanded)
# class StateLight(object):
#     def __init__(self):
#         pass

# action
class Action(object):
    def __init__(self, score):
        self.score = score

    def score(self):
        return self.score

# light-weighed actions? (nope, rely on specific representations)
# class ActionLight(object):
#     def __init__(self):
#         pass

# the searching graph (tracking the relations between states)
class SearchGraph(object):
    pass

# the scorer who manages the scoring and interacts with the model
class Scorer(object):
    pass

# the searcher
class Searcher(object):
    pass
