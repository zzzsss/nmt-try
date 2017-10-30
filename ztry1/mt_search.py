# first write a full process for standard beam-decoding, then use it as a template to see how to re-factor

# the main searching processes
from zl.search import State, Action, Searcher, SearchGraph, Losser, Scorer
from zl.model import Model
from zl import utils, layers
from collections import Iterable
import numpy

# class MTState(State):
#     def __init__(self, sg=None, action=None, prev=None, values=None):
#         super(MTState, self).__init__(sg, action, prev, values)
#
# class MTAction(Action):
#     def __init__(self, action_code, score):
#         super(MTAction, self).__init__(action_code, score)
MTState = State
MTAction = Action

# fixed possible candidates for NMT: all possible words in vocabularies
# class MTStateCands():
#     def __init__(self, state):
#         self.state = state
#
#     def get_state(self):
#         return self.state

# the scores for MTStateCands
# class MTScores:
#     pass

# Stated scorer, data stored as list(batch)//dict(state-id)
# -- simply, the stored data include the mapping of state-id -> calculation needed
# -- created once each searching process
class MTScorer(Scorer):
    def __init__(self, models, training):
        super(MTScorer, self).__init__()
        if not isinstance(models, Iterable):
            models = [models]
        self.models = models
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
        self.caches_asizes = []     # num of batches: sum(self.baches_bsizes)

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
    def rearrange_do(self, prev_idxes, mm, ret_names=None):
        # collect the groups and preserving flags
        groups = []     # (idx, [batch-idxes])
        accu = []  # (?exact, ?beam-based)
        prev_i0, next_beam_bar, next_beam_i = -1, 0, 0    # for BI
        for i0, i1 in prev_idxes:
            if i0 != prev_i0:
                # new idx
                groups.append((i0, [i1]))
                accu.append((0, 0))
                prev_i0 = i0
                next_beam_bar, next_beam_i = 0, 0
            else:
                groups[-1][1].append(i1)
            # check cont
            accu[-1][0] = i1+1 if i1==accu[-1][0] else -1   # if not seq, cannot accumulate up
            if i1 >= next_beam_bar:
                if accu[-1][1] != next_beam_bar:
                    accu[-1][1] = utils.Constants.MIN_V
                next_beam_bar += self.caches_bsizes[i0][next_beam_i]
                next_beam_i += 1
            accu[-1][1] = accu[-1][1]+1 if i1<next_beam_bar else utils.Constants.MIN_V
        # re-arrange them
        ret = {}
        for names, which in zip([mm.names_bv, mm.names_bi], [0, 1]):
            for n in names:
                if ret_names is not None and not n in ret_names:
                    continue
                ll = []
                for ac, g in zip(accu, groups):
                    ii = g[0]
                    cc = self.caches[ii]
                    if ac[which] == self.caches_asizes[ii]:
                        ll.append(cc)
                    else:
                        ll.append(layers.BK.batch_rearrange_one(cc, g[1]))
                ret[n] = layers.BK.concatenate_to_batch(ll)
        return ret

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
            self.caches_asizes.append(sum(cur_bsizes))
            for ii, one in enumerate(cur_flats):
                self.indexes[i][one.id] = (idx, ii)
        # average and return, returning the same structure as state_cands (list-batch of list-beam of list-candidates)
        # todo: assuming "results" as  the result key
        expr_results = [c["results"] for c in ccs]
        results = self._average_results(expr_results, cur_bsizes)
        return results

    def _average_results(self, expr_results, cur_bsizes):
        avg_e = layers.BK.average(expr_results)
        dims = layers.BK.dims(avg_e)
        utils.zcheck(len(dims)==1, "Wrong result dimensions.")
        shape = (layers.BK.bsize(avg_e), dims[0])
        avg_a = numpy.array(avg_e.value()).reshape(shape)
        ret = []
        cur_start = 0
        for ll in cur_bsizes:
            ret.append(avg_a[cur_start:cur_start+ll])
            cur_start += ll
        utils.zcheck(shape[0]==cur_start, "Wrong batch&beam shapes.")
        return ret

    # obtain result expressions --- return as batched ones
    def get_result_expressions(self, state_cands, rets, skip_rearrange=False):
        cur_states = state_cands        # for simplicity
        cur_flats = [s for s in utils.Helper.stream_rec(cur_states)]
        cur_ids = [x.id for x in cur_flats]
        # prepare the caches
        values = []
        for i, _mm in enumerate(self.models):
            # check and rearrange
            cur_idxes = [self.indexes[i][x] for x in cur_ids]    # list of (idx, batch-idx)
            could_skip = self.rearrange_check(cur_idxes)
            if skip_rearrange:
                utils.zcheck(could_skip, "Not continuous but skipping rearrange!")
            if not could_skip:
                cur_val = self.rearrange_do(cur_idxes, _mm, ret_names=rets)
            else:
                cur_val = self.caches[cur_idxes[0][0]]
            values.append(cur_val)
        # average the values
        values_avg = []
        for n in rets:
            vv = layers.BK.average([_v[n] for _v in values_avg])
            values_avg.append(vv)
        return values_avg

class MTSearcher(Searcher):
    # def __init__(self):
    #     # some records
    #     pass

    # todo: with several pruning and length checking
    @staticmethod
    def search_beam(models, length_normer, insts, target_dict, opts):
        # input: scorer, list of instances
        xs = [i[0] for i in insts]
        # ys = [i[1] for i in insts]
        sc = MTScorer(models, False)
        # init the search
        bsize = len(xs)
        opens = [[MTState(sg=SearchGraph())] for _ in range(bsize)]     # with only one init state
        ends = [[] for _ in range(bsize)]
        # search for them
        esize_all = opts["beam_size"]
        esize_one = opts["beam_size"]
        eos_code = target_dict.eos
        decode_maxlen = opts["decode_len"]
        for _ in range(decode_maxlen):
            # expand and generate cands
            results = sc.calc_step(opens, xs)    # list of list of list of scores
            next_opens = []
            for one_inst, one_end, one_res in zip(opens, ends, results):
                # for one instance
                extended_candidates = []
                # pruning locally
                for one_state, one_score in zip(one_inst, one_res):
                    top_cands = numpy.argpartition(one_score, max(-len(one_score), -esize_one))[-esize_one:]
                    for cand_action in top_cands:
                        code = int(cand_action)
                        prob = one_score[code]
                        next_one = MTState(prev=one_state, action=MTAction(code, numpy.log(prob), prob=prob))
                        extended_candidates.append(next_one)
                # sort globally
                best_cands = sorted(extended_candidates, key=lambda x: x.score_partial, reverse=True)[:esize_all]
                # pruning globally & set up
                # todo(warn): decrease beam size here
                next_opens_one = []
                cap = max(0, esize_all-len(one_end))
                for j in range(cap):
                    one_cand = best_cands[j]
                    if int(one_cand.action) == eos_code:
                        one_cand.mark_end()
                        one_end.append(one_cand)
                    else:
                        next_opens_one.append(one_cand)
                next_opens.append(next_opens_one)
            opens = next_opens
        # finish up the un-finished (maybe +1 step)
        results = sc.calc_step(opens, xs)
        for one_inst, one_end, one_res in zip(opens, ends, results):
            for one_state, one_score in zip(one_inst, one_res):
                code = eos_code
                prob = one_score[code]
                finished_one = MTState(prev=one_state, action=MTAction(code, numpy.log(prob), prob=prob))
                finished_one.mark_end()
                one_end.append(finished_one)
        # re-ranking to the final ones
        length_normer(utils.Helper.stream_rec(ends))
        final_list = [sorted(beam, key=lambda x: x.score_final, reverse=True) for beam in ends]
        return final_list

    # force gold tracking
    @staticmethod
    def _gen_y_step(ys, i, bsize):
        _mask = [1. if i<len(_y) else 0. for _y in ys]
        ystep = [_y[i] if i<len(_y) else 0 for _y in ys]
        mask_expr = layers.BK.inputVector(_mask)
        mask_expr = layers.BK.reshape(mask_expr, (1, ), bsize)
        return ystep, mask_expr

    @staticmethod
    def _mle_loss_step(probs, scores_exprs, ystep, mask_expr):
        one_loss = layers.BK.pickneglogsoftmax_batch(scores_exprs, ystep)
        one_loss = one_loss * mask_expr
        return one_loss

    @staticmethod
    def fb_loss(models, insts, backward):
        xs = [i[0] for i in insts]
        ys = [i[1] for i in insts]
        sc = MTScorer(models, True)
        bsize = len(xs)
        opens = [[MTState(sg=SearchGraph())] for _ in range(bsize)]     # with only one init state
        cur_maxlen = max([len(_y) for _y in ys])
        losses = []
        for i in range(cur_maxlen):
            results = sc.calc_step(opens, xs, skip_rearrange=True)
            scores_exprs = sc.get_result_expressions(opens, ["out_s"], skip_rearrange=True)[0]
            ystep, mask_expr = MTSearcher._gen_y_step(ys, i, bsize)
            loss = MTSearcher._mle_loss_step(results, scores_exprs, ystep, mask_expr)
            losses.append(loss)
        # -- final
        loss = layers.BK.esum(losses)
        loss = layers.BK.sum_batches(loss) / bsize
        loss_val = loss.value()
        if backward:
            loss.backward()
        return loss_val*bsize
