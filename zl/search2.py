# some searching routines
from . import search, utils
from .search import State, SearchGraph

# extract n-best list from a search graph
def extract_nbest(sg, n, length_reward=0., normalizing_alpha=0.):
    # length_reward could be accurate, but normalizing again brings approximity
    utils.zcheck(length_reward*normalizing_alpha==0., "It is meaningless to set them both!!")
    _sort_k = lambda one: (one.score_partial + length_reward*one.length) / (one.length ** normalizing_alpha)
    _TMP_KEY = "_extra_nbest"
    #
    cands = []
    for one in sg.get_ends():
        _set_recusively_one(one, sg, n, _sort_k)
        vs = one.get(_TMP_KEY)
        cands += vs
    v = sorted(cands, key=_sort_k, reversed=True)[:n]
    return v

def _set_recusively_one(one, sg, n, sort_k):
    _TMP_KEY = "_extra_nbest"
    _PRUNE_KEY = "PRUNING_LIST"
    utils.zcheck(not sg.is_pruned(one), "Not scientific to call on pruned states!!")
    if one.get(_TMP_KEY) is None:
        if one.is_start():
            # rebuild without search-graph
            v = [State(sg=None)]
            one.set(_TMP_KEY, v)
        else:
            combined_list = one.get(_PRUNE_KEY)
            if combined_list is None:
                combined_list = []
            utils.zcheck_ff_iter(combined_list, lambda x: x.get(_TMP_KEY) is None, "Not scientific pruning states!!")
            combined_list.append(one)
            # recursive with their prevs
            cands = []
            for pp in combined_list:
                _set_recusively_one(pp.prev, sg, n, sort_k)
                for one_prev in pp.prev.get(_TMP_KEY):
                    new_state = State(prev=one_prev, action=pp.action)
                    pp.transfer_values(new_state)
                    cands.append(new_state)
            # and combining (could be more effective using HEAP or sth)
            v = sorted(cands, key=sort_k, reversed=True)[:n]
            one.set(_TMP_KEY, v)    # not on pruned states
