# first write a full process for standard beam-decoding, then use it as a template to see how to re-factor

# the main searching processes
from zl.search import State, Action, SearchGraph
from zl.model import Model
from zl import utils, data
from . import mt_layers as layers
import numpy as np

MTState = State
MTAction = Action

def search_init():
    State.reset_id()

# a padding version of greedy search
# todo(warn) normer is not used when greedy decoding
def search_greedy(models, insts, target_dict, opts, normer):
    xs = [i[0] for i in insts]
    Model.new_graph()
    for _m in models:
        _m.refresh(False)
    bsize, finish_size = len(xs), 0
    opens = [State(sg=SearchGraph()) for _ in range(bsize)]
    ends = [None for _ in range(bsize)]
    yprev = [-1 for _ in range(bsize)]
    decode_maxlen = opts["decode_len"]
    caches = []
    eos_id = target_dict.eos
    for step in range(decode_maxlen+1):
        one_cache = []
        for mi, _m in enumerate(models):
            if step==0:
                cc = _m.start(xs)
            else:
                cc = _m.step(caches[-1][mi], yprev)
            one_cache.append(cc)
        caches.append(one_cache)
        # prepare next steps
        results = layers.BK.average([one["results"] for one in one_cache])
        results_v = results.npvalue().reshape((layers.BK.dims(results)[0], bsize)).T
        force_end = step >= decode_maxlen
        for j in range(bsize):
            rr = results_v[j]
            if ends[j] is None:
                next_y = eos_id
                if not force_end:
                    next_y = int(rr.argmax())
                score = np.log(rr[next_y])
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
    # return them
    return [[s] for s in ends]

def nargmax(v, n):
    # return UNORDERED list of (id, value)
    thres = max(-len(v), -n)
    ids = np.argpartition(v, thres)[thres:]
    return [int(i) for i in ids]

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

def search_beam(models, insts, target_dict, opts, normer):
    xs = [i[0] for i in insts]
    Model.new_graph()
    for _m in models:
        _m.refresh(False)
    bsize = len(xs)
    esize_all = opts["beam_size"]
    esize_one = opts["beam_size"]
    decode_maxlen = opts["decode_len"]
    remain_sizes = [esize_all for _ in range(bsize)]
    opens = [[State(sg=SearchGraph())] for _ in range(bsize)]
    ends = [[] for _ in range(bsize)]
    bh = BatchedHelper(opens)
    caches = []
    eos_id = target_dict.eos
    yprev = None
    for step in range(decode_maxlen+1):
        one_cache = []
        if step==0:
            for mi, _m in enumerate(models):
                cc = _m.start(xs)
                one_cache.append(cc)
            # todo(warn) init normer and pruner here!! (after the first step)
            # todo(warn) only using the first model
            pred_lens = models[0].predict_length(insts, cc=one_cache[0])     # (#insts, ) of real lengths
        else:
            for mi, _m in enumerate(models):
                cc = _m.step(caches[-1][mi], yprev)
                one_cache.append(cc)
        # select cands
        results = layers.BK.average([one["results"] for one in one_cache])
        results_v0 = results.npvalue()
        results_v = results_v0.reshape(layers.BK.dims(results)[0], layers.BK.bsize(results)).T
        nexts = []
        force_end = step >= decode_maxlen
        for i in range(bsize):
            # for each one in the batch
            r_start = bh.get_basis(i)
            prev_states = opens[i]
            cur_cands = []
            for j, one in enumerate(prev_states):
                # for each one in the beam of one instance
                one_result = results_v[r_start+j]
                # todo(warn): force end for the last step
                one_cands = [eos_id] if force_end else nargmax(one_result, esize_one)
                for idx in one_cands:
                    # todo(warn): (prev-inner_dix)
                    cur_cands.append((j, State(prev=prev_states[j], action=Action(idx, np.log(one_result[idx])))))
            # sorting them all
            cur_cands.sort(key=(lambda x: x[-1].score_partial), reverse=True)
            # append the next cands
            cur_cands = cur_cands[:remain_sizes[i]]
            # prepare for next steps
            cur_nexts = []
            opens[i] = []
            for prev_inner_idx, new_state in cur_cands:
                action_id = new_state.action_code
                if action_id == eos_id:
                    ends[i].append(new_state)
                    new_state.mark_end()
                    remain_sizes[i] -= 1
                else:
                    opens[i].append(new_state)
                    cur_nexts.append((prev_inner_idx, action_id))
            nexts.append(cur_nexts)
        # re-arrange for next step if not ending
        if sum(remain_sizes) <= 0:
            break
        bv_orders, bi_orders, new_ys = bh.rerange(nexts)
        new_caches = [_m.rerange(_c, bv_orders, bi_orders) for _c, _m in zip(one_cache, models)]
        caches.append(new_caches)
        yprev = new_ys
    # final re-ranking
    normer(ends, pred_lens)
    final_list = [sorted(beam, key=lambda x: x.score_partial, reverse=True) for beam in ends]
    # data.Vocab.i2w(target_dict, final_list[0][0])
    # final_list[0][0].sg.print_graph(target_dict)
    return final_list
