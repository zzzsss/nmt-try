# first write a full process for standard beam-decoding, then use it as a template to see how to re-factor

# the main searching processes
from zl.search import State, Action, Searcher, SearchGraph, Results, Scorer
from zl.model import Model
from zl import utils, layers
from collections import Iterable
import numpy

MTState = State
MTAction = Action

# a padding version of greedy search
def search_greedy(models, insts, target_dict, opts):
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
                score = rr[next_y]
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

# class MTSearcher(Searcher):
#     # todo: with several pruning and length checking
#     @staticmethod
#     def search_beam(models, insts, target_dict, opts):
#         # input: scorer, list of instances
#         xs = [i[0] for i in insts]
#         # ys = [i[1] for i in insts]
#         sc = MTScorer(models, False)
#         # init the search
#         bsize = len(xs)
#         opens = [[MTState(sg=SearchGraph())] for _ in range(bsize)]     # with only one init state
#         ends = [[] for _ in range(bsize)]
#         # search for them
#         esize_all = opts["beam_size"]
#         esize_one = opts["beam_size"]
#         eos_code = target_dict.eos
#         decode_maxlen = opts["decode_len"]
#         still_going = False
#         for _ in range(decode_maxlen):
#             # expand and generate cands
#             sc.calc_step(opens, xs)
#             results = sc.get_result_values(opens, ["results"])[0]    # list of list of list of scores
#             next_opens = []
#             still_going = False
#             for j in range(bsize):
#                 one_inst, one_end, one_res = opens[j], ends[j], results[j]
#                 # for one instance
#                 extended_candidates = []
#                 # pruning locally
#                 for i in range(len(one_inst)):
#                     one_state, one_score = one_inst[i], one_res[i]
#                     top_cands = one_score.argmax(esize_one)
#                     for cand_action in top_cands:
#                         code = int(cand_action)
#                         prob = one_score[code]
#                         next_one = MTState(prev=one_state, action=MTAction(code, numpy.log(prob), prob=prob))
#                         extended_candidates.append(next_one)
#                 # sort globally
#                 best_cands = sorted(extended_candidates, key=lambda x: x.score_partial, reverse=True)[:esize_all]
#                 # pruning globally & set up
#                 # todo(warn): decrease beam size here
#                 next_opens_one = []
#                 cap = max(0, esize_all-len(one_end))
#                 for j in range(cap):
#                     one_cand = best_cands[j]
#                     if int(one_cand.action) == eos_code:
#                         one_cand.mark_end()
#                         one_end.append(one_cand)
#                     else:
#                         next_opens_one.append(one_cand)
#                 next_opens.append(next_opens_one)
#                 cap = max(0, esize_all-len(one_end))
#                 still_going = still_going or (cap>0)
#             opens = next_opens
#             if not still_going:
#                 break
#         # finish up the un-finished (maybe +1 step)
#         if still_going:
#             sc.calc_step(opens, xs)
#             results = sc.get_result_values(opens, ["results"])[0]
#             for j in range(bsize):
#                 one_inst, one_end, one_res = opens[j], ends[j], results[j]
#                 for i in range(len(one_inst)):
#                     one_state, one_score = one_inst[i], one_res[i]
#                     code = eos_code
#                     prob = one_score[code]
#                     finished_one = MTState(prev=one_state, action=MTAction(code, numpy.log(prob), prob=prob))
#                     finished_one.mark_end()
#                     one_end.append(finished_one)
#         # re-ranking to the final ones
#         # todo: length normer: to final_score
#         final_list = [sorted(beam, key=lambda x: x.score_partial, reverse=True) for beam in ends]
#         return final_list
