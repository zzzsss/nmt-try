# first write a full process for standard beam-decoding, then use it as a template to see how to re-factor

# the main searching processes
from zl.search import State, Action, SearchGraph
from zl.search2 import extract_nbest
from zl.model import Model
from zl import utils, data
from . import mt_layers as layers
import numpy as np
from collections import defaultdict

MTState = State
MTAction = Action

def search_init():
    State.reset_id()

# a padding version of greedy search
# todo(warn) normers and pruners are not used for this greedy decoding (mostly used for dev)
def search_greedy(models, insts, target_dict, opts, normer):
    xs = [i[0] for i in insts]
    Model.new_graph()
    for _m in models:
        _m.refresh(False)
    _m0 = models[0]     # maybe common for all models, todo(warn) this inhibit certain varieties of model diff
    bsize, finish_size = len(xs), 0
    opens = [State(sg=SearchGraph()) for _ in range(bsize)]
    ends = [None for _ in range(bsize)]
    yprev = [-1 for _ in range(bsize)]
    decode_maxlen = opts["decode_len"]
    decode_maxratio = opts["decode_ratio"]
    caches = []
    eos_id = target_dict.eos
    for step in range(decode_maxlen):
        one_cache = []
        if step==0:
            for mi, _m in enumerate(models):
                cc = _m.start(xs)
                one_cache.append(cc)
        else:
            for mi, _m in enumerate(models):
                cc = _m.step(caches[-1][mi], yprev)
                one_cache.append(cc)
        caches.append(one_cache)
        # prepare next steps
        results = layers.BK.average([one["results"] for one in one_cache])
        results_v0 = layers.BK.get_value_np(results)
        results_v = results_v0.reshape((layers.BK.dims(results)[0], bsize)).T
        for j in range(bsize):
            rr = results_v[j]
            if ends[j] is None:
                # cur off if one of the criteria says it is time to end.
                # todo(warn): upper length pruner
                force_end = ((step+1) >= min(decode_maxlen, len(xs[j])*decode_maxratio))
                next_y = eos_id
                if not force_end:
                    next_y = int(rr.argmax())
                score = _m0.explain_result(rr[next_y])
                # check eos
                if next_y == eos_id:
                    ends[j] = State(prev=opens[j], action=Action(next_y, score))
                    ends[j].mark_end()
                    finish_size += 1
                else:
                    opens[j] = State(prev=opens[j], action=Action(next_y, score))
            else:
                next_y = 0
            yprev[j] = next_y
        if finish_size >= bsize:
            break
    # return them
    return [[s] for s in ends]

def nargmax(v, n):
    # return ORDERED list of (id, value)
    thres = max(-len(v), -n)
    ids = np.argpartition(v, thres)[thres:]
    ret = sorted([int(i) for i in ids], key=lambda x: v[x], reverse=True)
    return ret

class BatchedHelper(object):
    def __init__(self, opens):
        # opens is list of list
        self.bsize = None
        self.shapes = None
        self.bases = None
        self._build(opens)

    def _build(self, opens):
        b = [0]
        for ss in opens:
            b.append(b[-1]+len(ss))
        self.bsize = len(opens)
        self.shapes = [len(one) for one in opens]
        self.bases = b

    def get_basis(self, i):
        return self.bases[i]

    def rerange(self, nexts):
        # nexts is list of list of (prev-idx, next_token), return orders, next_ys
        utils.zcheck_matched_length(nexts, self.shapes)
        bv_orders = []
        new_ys = []
        break_bv, break_bi = False, False
        for i in range(len(nexts)):
            nns = nexts[i]
            bas = self.bases[i]
            if len(nns) != self.shapes[i]:
                break_bi = True
            for one in nns:
                j = one[0]
                if (len(bv_orders)==0 and bas+j!=0) or (len(bv_orders)>0 and bas+j-1!=bv_orders[-1]):
                    break_bv = True
                bv_orders.append(bas+j)
                new_ys.append(one[1])
        if len(bv_orders) != self.bases[-1]:
            break_bv, break_bi = True, True
        # rebuild
        self._build(nexts)
        # return
        if break_bi:
            return bv_orders, bv_orders, new_ys
        elif break_bv:
            return bv_orders, None, new_ys
        else:
            return None, None, new_ys

# synchronized with y-steps
def search_beam(models, insts, target_dict, opts, normer):
    xs = [i[0] for i in insts]
    xwords = [i.get_origin(0) for i in insts]
    Model.new_graph()
    for _m in models:
        _m.refresh(False)
    _m0 = models[0]     # maybe common for all models, todo(warn) this inhibit certain varieties of model diff
    bsize = len(xs)
    esize_all = opts["beam_size"]
    decode_maxlen = opts["decode_len"]
    decode_maxratio = opts["decode_ratio"]
    # if need to get att-weights (todo(warn))
    need_att = opts["decode_replace_unk"]
    # pruners
    pr_local_expand = opts["pr_local_expand"]
    pr_local_diff = opts["pr_local_diff"]
    pr_local_penalty = opts["pr_local_penalty"]
    #
    pr_global_expand = opts["pr_global_expand"]
    pr_global_diff = opts["pr_global_diff"]
    pr_global_penalty = opts["pr_global_penalty"]
    pr_tngram_n = opts["pr_tngram_n"]
    pr_tngram_range = opts["pr_tngram_range"]
    #
    remain_sizes = [esize_all for _ in range(bsize)]
    opens = [[State(sg=SearchGraph())] for _ in range(bsize)]
    ends = [[] for _ in range(bsize)]
    bh = BatchedHelper(opens)
    caches = []
    eos_id = target_dict.eos
    yprev = None
    for step in range(decode_maxlen):
        one_cache = []
        if step==0:
            for mi, _m in enumerate(models):
                cc = _m.start(xs)
                one_cache.append(cc)
            # todo(warn) init normer and pruner here!! (after the first step) # (#insts, ) of real lengths
            pred_lens = np.average([_m.predict_length(insts, cc=_c) for _m, _c in zip(models, one_cache)], axis=0)
            pred_lens_sigma = np.average([_m.lg.get_real_sigma() for _m in models], axis=0)
            pr_len_upper = pred_lens + opts["pr_len_khigh"] * pred_lens_sigma
            pr_len_lower = pred_lens - opts["pr_len_klow"] * pred_lens_sigma
        else:
            for mi, _m in enumerate(models):
                cc = _m.step(caches[-1][mi], yprev)
                one_cache.append(cc)
        # select cands
        results = layers.BK.average([one["results"] for one in one_cache])
        this_bsize = layers.BK.bsize(results)
        results_v0 = layers.BK.get_value_np(results)
        results_v1 = results_v0.reshape(-1, this_bsize)
        results_v = results_v1.T
        if need_att:
            atts_e = layers.BK.average([one["att"] for one in one_cache])
            atts_v = layers.BK.get_value_np(atts_e).reshape(-1, this_bsize).T
        else:
            atts_v = [None for _ in range(this_bsize)]
        nexts = []
        for i in range(bsize):
            # cur off if one of the criteria says it is time to end.
            # todo(warn): upper length pruner
            force_end = ((step+1) >= min(decode_maxlen, pr_len_upper[i], len(xs[i])*decode_maxratio))
            # for each one in the batch
            r_start = bh.get_basis(i)
            prev_states = opens[i]
            cur_cands = []
            cur_xsrc = xwords[i] if need_att else None
            for j, one in enumerate(prev_states):
                # for each one in the beam of one instance
                one_result = results_v[r_start+j]
                one_attv = atts_v[r_start+j]
                # todo(warn): force end for the last step
                one_cands = [eos_id] if force_end else nargmax(one_result, esize_all)
                # local pruning
                cand_states = [State(prev=prev_states[j], action=Action(idx, _m0.explain_result(one_result[idx])),
                                      attention_weights=one_attv, attention_src=cur_xsrc, _tmp_prev_idx=j) for idx in one_cands]
                survive_local_cands = Pruner.local_prune(cand_states, pr_local_expand, pr_local_diff, pr_local_penalty)
                cur_cands += survive_local_cands
            # sorting them all
            cur_cands.sort(key=(lambda x: x.score_partial), reverse=True)
            # global pruning
            ok_cands = Pruner.global_prune_ngram_greedy(cand_states=cur_cands, rest_beam_size=remain_sizes[i], sig_beam_size=pr_global_expand, thresh=pr_global_diff, penalty=pr_global_penalty, ngram_n=pr_tngram_n, ngram_range=pr_tngram_range)
            # prepare for next steps
            cur_nexts = []
            opens[i] = []
            pruned_ends = []        # the ended ones that are pruned away
            for new_state in ok_cands:
                prev_inner_idx = new_state.get("_tmp_prev_idx")
                action_id = new_state.action_code
                if action_id == eos_id:
                    new_state.mark_end()
                    # todo: pruning short ones
                    if new_state.length <= pr_len_lower[i]:
                        new_state.set("PR_END", True)
                        pruned_ends.append(new_state)
                    else:
                        new_state.set("FINISH", True)
                        ends[i].append(new_state)
                        remain_sizes[i] -= 1
                else:
                    new_state.set("EXPAND", True)
                    opens[i].append(new_state)
                    cur_nexts.append((prev_inner_idx, action_id))
            # what if we pruned away all of them -> add them back even though they are pruned
            if len(ends[i])==0 and len(cur_nexts)==0:
                for one in pruned_ends:
                    one.set("PR_END", False)
                    ends[i].append(one)
                    remain_sizes[i] -= 1
            # append for one inst
            nexts.append(cur_nexts)
        # re-arrange for next step if not ending
        if sum(remain_sizes) <= 0:
            break
        bv_orders, bi_orders, new_ys = bh.rerange(nexts)
        new_caches = [_m.rerange(_c, bv_orders, bi_orders) for _c, _m in zip(one_cache, models)]
        caches.append(new_caches)
        yprev = new_ys
    # final re-ranking
    if opts["decode_latnbest"]:
        ends = [extract_nbest(ol[0].sg, esize_all, length_reward=opts["decode_latnbest_lreward"], normalizing_alpha=opts["decode_latnbest_nalpha"]) for ol in ends]
    normer(ends, pred_lens)
    final_list = [sorted(beam, key=lambda x: x.score_final, reverse=True) for beam in ends]
    # data.Vocab.i2w(target_dict, final_list[0][0])
    # final_list[0][0].sg.show_graph(target_dict)
    return final_list

class Pruner(object):
    @staticmethod
    def local_prune(cand_states, maxsize, thresh, penalty):
        # on sorted list
        # -- always add the max-scoring one
        ret = [cand_states[0]]
        one_score_max = cand_states[0].action_score()
        # -- possibly pruning the rest (local pruning)
        for i, one in enumerate(cand_states[1:], start=1):
            one_score_cur = one.action_score()
            if i >= maxsize:
                one.set("PR_LOCAL_EXPAND", True)
            elif one_score_cur <= one_score_max - thresh:
                one.set("PR_LOCAL_DIFF", True)
            else:
                one_score_cur -= i * penalty    # diversity penalty
                one.action_score(one_score_cur) # penalty
                ret.append(one)
        return ret

    @staticmethod
    def _state_ngram(one, ng_n, ng_range):
        actions = one.get_path("action_code")
        histories = one.get_path()
        ret = []      # (sig, state)
        length = len(actions)
        for i in range(ng_range):
            # at least n-gram, no <bos>
            if length-i < ng_n:
                break
            ac_list = actions[-ng_n-i:-1-i]
            st = histories[-1-i]
            ret.append(("|".join([str(_one for _one in ac_list)]), st))
        return ret

    @staticmethod
    def global_prune_ngram_greedy(cand_states, rest_beam_size, sig_beam_size, thresh, penalty, ngram_n, ngram_range):
        # on sorted list, comparing according to normalized scores -- greedy pruning
        # todo: how could we compare diff length states? -> normalize partial score
        # todo: how to do pruning and sig-max (there might be crossings)? -> take the greedy way
        _get_score_f = (lambda x: x.score_partial/x.length)
        sig_ngram_maxs = {}     # all-step max (for survived ones)
        sig_ngram_curnum = defaultdict(int)     # last-step state lists for sig
        sig_ngram_allnum = defaultdict(int)     # all-step state counts for sig
        temp_ret = []
        # pruning
        for one in cand_states:
            if len(temp_ret) >= rest_beam_size:
                one.set("PR_BEAM", True)
                continue
            if ngram_range <= 0:
                # sig pruning off
                temp_ret.append(one)
                continue
            # ngram sigs, according to the listing
            them = Pruner._state_ngram(one, ngram_n, ngram_range)
            if len(them) > 0:
                this_pruned = False
                # pruning according to sig-size and thresh
                cur_sig, state = them[-1]
                one.set("SIG_NGRAM", cur_sig)
                this_score, high_score = _get_score_f(state), _get_score_f(sig_ngram_maxs[cur_sig])
                if cur_sig in sig_ngram_maxs:
                    if this_score <= high_score:
                        if sig_ngram_curnum[cur_sig] >= sig_beam_size:  # sig_beam_size could be zero
                            one.set("PR_NGRAM_EXPAND", True)
                            this_pruned = True
                        elif this_score <= high_score - thresh:
                            one.set("PR_NGRAM_DIFF", True)
                            this_pruned = True
                # adding
                if not this_pruned:
                    # add all steps for this one
                    for one_sig, one_state in them:
                        if one_sig not in sig_ngram_maxs or _get_score_f(one_state) > _get_score_f(sig_ngram_maxs[one_sig]):
                            sig_ngram_maxs[one_sig] = one_state
                        sig_ngram_allnum[one_sig] += 1
                    sig_ngram_curnum[cur_sig] += 1      # only last step
                    temp_ret.append(state)
                else:
                    # set pruners and record in the sg
                    pruner_one = sig_ngram_maxs[cur_sig]
                    one.set("PR_PRUNER", pruner_one)
                    pruner_one.add_list("PRUNING_LIST", one)
            else:
                # for example, the first several steps
                temp_ret.append(one)
        # penalizing
        sig_ngram_finalnum = defaultdict(int)
        ret = []
        for one in temp_ret:
            one_sig = one.get("SIG_NGRAM")
            if one_sig is None:
                ret.append(one)
            else:
                # count prev steps if not ranking the first one, but ignore in other cases
                if sig_ngram_finalnum[one_sig] == 0 and _get_score_f(one) < _get_score_f(sig_ngram_maxs[one_sig]):
                    sig_ngram_finalnum[one_sig] += 1
                one_score_cur = one.action_score() - sig_ngram_finalnum[one_sig] * penalty    # diversity penalty
                one.action_score(one_score_cur)
                sig_ngram_finalnum[one_sig] += 1
        return ret

    # @staticmethod
    # def global_prune_ngram_all(cand_states, rest_beam_size, sig_beam_size, thresh, penalty, ngram_n, ngram_range):
    #     # todo(warn): how to deal with cycles of pruning
    #     raise NotImplementedError()

def search_branch(models, insts, target_dict, opts, normer):
    pass
