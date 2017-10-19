# first write a full process for standard beam-decoding, then use it as a template to see how to re-factor

# the main searching processes
from zl.search import State, StateLight, Searcher, SearchGraph
from zl.model import Model
from collections import Iterable
from zl import utils
import numpy

class MTState(State):
    def __init__(self, sg, action=None, prev=None, values=None):
        super(MTState, self).__init__(sg, action, prev, values)

class MTStateLight(StateLight):
    def __init__(self, prev, action):
        self.prev = prev
        self.action = action

class MTAction:
    def __init__(self):
        pass

# Stated scorer, data stored as list(batch)//dict(state-id)
# -- simply, the stored data include the mapping of state-id -> calculation needed
class MTScorer:
    def __init__(self, models):
        if not isinstance(models, Iterable):
            models = [models]
        self.models = models
        # init
        Model.init_graph()
        for _mm in models:
            _mm.refresh(False, False)
        # caches


    def prepare(self, xs):
        # pre-calculations for some values for xs (encoder part) and init for y (start of decode part)
        pass

    def expand(self, states):
        # calculate for the states and generate the scores for candidates
        return []

class MTSearcher(Searcher):
    # (todo) this is in fact the agenda, firstly a standard beam searcher
    def search(self, models, xs, opts):
        if not isinstance(xs, Iterable):
            xs = [xs]
        # init the scorer
        sc = MTScorer(models)
        # init the search
        bsize = len(xs)
        sg = SearchGraph()
        opens = [[MTState(sg)] for _ in range(bsize)]
        ends = [[] for _ in range(bsize)]
        sc.prepare(xs)
        # search for them
        esize_all = 8
        esize_one = 8
        while True:
            # expand and generate cands
            results = sc.expand(opens)    # list of list of list of scores
            # select cands
            utils.zcheck_matched_length(opens, results)
            cands = []
            for one_op, one_res in zip(opens, results):
                candidates = []
                for one_state, one_score in zip(one_op, one_res):
                    sc = one_score.scores()
                    top_ids = numpy.argpartition(sc, max(-len(sc),-esize_one))[-esize_one:]
                    for cand_action in top_ids:
                        next_one = int(cand_action)
                        candidates.append(MTStateLight(one_state, next_one))
                best_cands = sorted(candidates, key=lambda x: x.score, reverse=True)[:esize_all]
                cands.append(best_cands)
            # setup the next states
            for i, one_cands in enumerate(cands):
                # generate the new opens
                opens[i] = None  # what?

