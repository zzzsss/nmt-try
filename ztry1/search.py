# the main searching processes
from zl.search import State, Action, Searcher

class MTState(State):
    def __init__(self, action):
        super(MTState, self).__init__()
        self.action = action

class MTAction(Action):
    def __init__(self):
        super(Action, self).__init__()

class MTSearcher(Searcher):
    pass
