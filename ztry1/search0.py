# first write a full process for standard beam-decoding, then use it as a template to see how to re-factor

# the main searching processes
from zl.search import State, Searcher, SearchGraph
from zl.model import Model
from zl import utils, layers
from collections import Iterable, defaultdict
import numpy

class MTState(State):
    def __init__(self, sg, action=None, prev=None, values=None):
        super(MTState, self).__init__(sg, action, prev, values)

class MTAction:
    def __init__(self, action_idx, masked=False):
        self._action = action_idx
        self.masked = masked

    def __int__(self):
        return self._action

# fixed possible candidates for NMT: all possible words in vocabularies
# class MTStateCands():
#     def __init__(self, state):
#         self.state = state
#
#     def get_state(self):
#         return self.state

# Stated scorer, data stored as list(batch)//dict(state-id)
# -- simply, the stored data include the mapping of state-id -> calculation needed
# -- created once each searching process
class MTScorer:
    def __init__(self, models, training, opts):
        if not isinstance(models, Iterable):
            models = [models]
        self.models = models
        self.opts = opts    # for the info of dropouts
        self.training = training
        # init
        Model.new_graph()
        for _mm in models:
            _mm.refresh(self.training)
        # caches: separated for each model, also for each element in the batches
        self.insts_num = -1
        # todo: with re-arrange, one value could appear in multiple places
        self.indexes = [{} for _ in range(len(models))]     # idx is: state-id => (cache-id, inside-id)
        self.caches = []            # expressions: dicts
        self.caches_bsizes = []     # the beam sizes of them, sum up to esizes: lists

    @property
    def started(self):
        return self.insts_num > 0

    def rearrange_check(self, idxes):
        # list of (idx, batch-idx), check continuous
        ii = idxes[0][0]
        for i, p in enumerate(idxes):
            if p[0]!=ii or p[1]!=i:
                return False
        return len(idxes)==sum(self.caches_bsizes[ii])

    # todo: could be more efficient by reordering
    def rearrange_do(self, prev_idxes, mm):
        # collect the groups
        groups = []     # (idx, [batch-idxes])
        prev_i0 = -1
        for i0, i1 in prev_idxes:
            if i0 != prev_i0:
                groups.append([i1])
                prev_i0 = i0
            else:
                groups[-1].append(i1)
        # prepare the idxes

        return {}

    def calc_step(self, state_cands, xs, skip_rearrange=False):
        # calculate for the states and generate the scores for candidates
        cur_states = state_cands        # for simplicity
        cur_flats = [s for s in utils.Helper.stream_rec(cur_states)]
        ccs = []
        if not self.started:
            self.insts_num = len(xs)
            repeat_time = len(state_cands[0])
            # assert that they are all initial states
            utils.zcheck_ff_iter(cur_states, lambda x: len(x)==repeat_time and all(y.is_start() for y in x),
                                 "Unexpected initial states.", _forced=True)
            utils.zcheck_matched_length(cur_states, xs, _forced=True)
            # start
            for _mm in self.models:
                cc = _mm.start(xs, repeat_time)
                ccs.append(cc)
        else:
            # assert that they are all not initial states
            utils.zcheck_ff_iter(cur_states, lambda x: all(not y.is_start() for y in x), "Unexpected initial states.")
            #
            prev_ids = [x.prev.id for x in cur_flats]
            next_actions = [int(x.action) for x in cur_flats]
            # calculate
            for i, _mm in enumerate(self.models):
                # check and rearrange
                prev_idxes = [self.indexes[i][x] for x in prev_ids]    # list of (idx, batch-idx)
                could_skip = self.rearrange_check(prev_idxes)
                if skip_rearrange:
                    utils.zcheck(could_skip, "Not continuous but skipping rearrange!")
                if not could_skip:
                    prev_val = self.rearrange_do(prev_idxes, _mm)
                else:
                    prev_val = self.caches[prev_idxes[0][0]]
                # actual calculate
                cc = _mm.step(prev_val, next_actions)
                ccs.append(cc)
        # add the caches
        cur_bsizes = [len(one) for one in cur_states]
        for i, cc in enumerate(ccs):
            idx = len(self.caches)
            self.caches.append(cc)
            self.caches_bsizes.append(cur_bsizes)
            for ii, one in enumerate(cur_flats):
                self.indexes[i][one.id] = (idx, ii)
        # average and return
        # TODO
        return

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
