# decoding with the model
import dynet as dy
import numpy as np
import utils, data

def _check_order(l):
    return (len(l)==0 and l[0]==0) or (l[-1]==l[-2]+1)

def decode(diter, mms, target_dict, opts, outf):
    one_recorder = utils.OnceRecorder("DECODE")
    with open(outf, "w") as f:
        for xs, _1, _2, _3 in diter:
            rs = search(xs, mms, opts, opts["decode_way"], opts["decode_batched"])
            one_recorder.record(xs, None, 0, 0)
            for r in rs:
                strs = data.Dict.i2w(target_dict, r)
                f.write(" ".join(strs)+"\n")
    one_recorder.report()

# the main searching routine
def search(xs, models, opts, strategy, batched):
    # xs: list(batch) of list(sent) of int
    if type(models) not in [list, tuple]:
        models = [models]
    dy.renew_cg()
    for _mm in models:
        _mm.refresh(False, False)   # no need for batch-size
    # prepare
    st = {"beam":BeamStrategy(opts["beam_size"]), "sample":SamplingStrategy(opts["sample_size"])}[strategy]
    PR = BatchedProcess if batched else NonBatchedProcess
    pr = PR(models, st.width())
    normer = Normer() if opts["normalize"] <= 0 else PolyNormer(opts["normalize"])
    hypos = [[Hypo(normer=normer) for _z in range(st.start_hypo_num())] for _ in range(len(xs))]
    finished = [[] for _ in range(len(xs))]
    # for max-steps
    nexts, orders = None, None
    eos_ind = models[0].target_dict.eos
    for s in range(opts["decode_len"]):
        # feed one and get results
        if s == 0:
            results = pr.start(xs)
        else:
            results = pr.feed(nexts, orders)
        # get next steps
        cands = st.select_nexts(hypos, results)
        new_nexts, new_orders, new_hypos = [[] for _ in range(len(xs))], [[] for _ in range(len(xs))], [[] for _ in range(len(xs))]
        ordering_flag = True
        for i, one_cands in enumerate(cands):
            for j, cc in enumerate(one_cands):
                this_hypo, this_action, prev_index = cc
                if this_action == eos_ind:
                    finished[i].append(this_hypo)
                else:
                    new_nexts[i].append(this_action)
                    new_orders[i].append(prev_index)
                    ordering_flag = _check_order(new_orders[i]) and ordering_flag
                    new_hypos[i].append(this_hypo)
        nexts, orders = new_nexts, (None if ordering_flag else new_orders)
        # test finished ?
        if all([len(fs) >= st.width() for fs in finished]):
            break
    # force eos
    final_results = pr.feed(nexts, orders)
    for i, hs in enumerate(hypos):
        for j, h in enumerate(hs):
            finished[i].append(Hypo(last_action=eos_ind, prev=h, score_one=final_results[i][j][eos_ind]))
    # final check and ordering
    rets = [[f.get_path() for f in sorted(fs, key=lambda x:x.final_score, reverse=True)[:st.width()]] for fs in finished]
    return rets

# length normer
class Normer(object):
    def norm_score(self, s, l):
        return s

class PolyNormer(Normer):
    def __init__(self, alpha):
        self.alpha = alpha

    def norm_score(self, s, l):
        return s/pow(l, self.alpha)

# hypothesis
class Hypo(object):
    DefaultNormer = Normer
    def __init__(self, last_action=None, prev=None, score_one=0., normer=None):
        self.last_action = last_action    # None for the start symbol
        self.prev = prev
        self.score_one = score_one
        self.score_acc = prev.score_acc+score_one if prev else 0.
        self.length = prev.length+1 if prev else 0
        self.normer = None
        if normer is not None:
            self.normer = normer
        elif self.prev is not None:
            self.normer = self.prev.normer
        else:
            self.normer = Hypo.DefaultNormer()

    @property
    def partial_score(self):
        return self.normer.norm_score(self.score_acc, self.length)

    @property
    def final_score(self):
        return self.normer.norm_score(self.score_acc, self.length)

    def get_path(self):
        if self.prev is None:
            return []
        else:
            l = self.prev.get_path()
            l.append(self.last_action)
            return l

# how the batch of decoding states are represented
class Process(object):
    def __init__(self, mms, expand):
        self.mms = mms
        self.expand = expand
        self.bsize = -1
        # list(mms) of ss
        self.hiddens = None

    @property
    def all_size(self):
        return self.expand*self.bsize

    def start(self, **argv):
        raise NotImplementedError()

    def feed(self, **argv):
        raise NotImplementedError()

class BatchedProcess(Process):
    def __init__(self, mms, expand):
        super(BatchedProcess, self).__init__(mms, expand)

    def _fold_list(self, ss):
        # return list of list of scores (bs, expand, vocab)
        return ss.reshape((self.bsize, self.expand, -1))

    def _flat_list(self, ll, change):
        r = []
        base = 0
        for l in ll:
            utils.DEBUG_check(len(l) == self.expand)
            r += [base+one for one in l]
            base += len(l)
        utils.DEBUG_check(base == self.all_size)
        return r

    def _pad_list(self, nexts, orders):
        # Warning, in-place appending
        utils.DEBUG_check(len(nexts) == len(orders))
        utils.DEBUG_check(len(nexts) == self.bsize)
        for i in range(self.bsize):
            utils.DEBUG_check(len(nexts[i]) == len(orders[i]))
            for _ in range(len(nexts[i]), self.expand):
                nexts[i].append(0)
                orders[i].append(0)

    def start(self, xs):
        self.bsize = len(xs)
        cur_hiddens = []
        cur_probs = []
        for _m in self.mms:
            ss = _m.prepare_enc(xs, self.expand)
            hi, at = ss.get_results_one()
            ye = _m.get_start_yembs(self.all_size)
            sc = _m.get_score(at, hi, ye)
            cur_hiddens.append(ss)
            cur_probs.append(dy.softmax(sc))
        self.hiddens = cur_hiddens
        # average scores
        final_score = cur_probs[0] if (len(cur_probs)==1) else dy.average(cur_probs)
        return self._fold_list(final_score.value())

    def feed(self, nexts, orders):
        # orders/nexts => list(batch) of list(expand) of int
        self._pad_list(nexts, orders)
        flat_nexts = self._flat_list(nexts, False)
        flat_orders = self._flat_list(orders, True) if orders is not None else None
        cur_hiddens = []
        cur_probs = []
        for i, _m in enumerate(self.mms):
            ye = _m.get_embeddings_step(flat_nexts, _m.embed_trg)
            if flat_orders is not None:     # no change if None (for eg., when sampling)
                self.hiddens[i].shuffle(flat_orders)
            ss = _m.dec.feed_one(self.hiddens[i], ye)
            hi, at = ss.get_results_one()
            sc = _m.get_score(at, hi, ye)
            cur_hiddens.append(ss)
            cur_probs.append(dy.softmax(sc))
        self.hiddens = cur_hiddens
        # average scores
        final_score = cur_probs[0] if len(cur_probs)==1 else dy.average(cur_probs)
        return self._fold_list(final_score.value())

class NonBatchedProcess(Process):
    def __init__(self, mms, expand):
        super(NonBatchedProcess, self).__init__(mms, expand)
        # hiddens: (batches, expands, models) #different orders from the Batched version#

    def _avg_probs(self, cur_probs):
        # input: (batches, expands, models) => output: (batches, expands)
        r = []
        for i, one in enumerate(cur_probs):
            r.append([])
            for j, ll in enumerate(one):
                sc = ll[0].value() if len(ll)==1 else dy.average(ll).value()
                r[-1].append(sc)
        return r

    def start(self, xs):
        self.bsize = len(xs)
        cur_hiddens = [[[] for _ in range(self.expand)] for _j in xs]
        cur_probs = [[[] for _ in range(self.expand)] for _j in xs]
        for j, one in enumerate(xs):
            for i, _m in enumerate(self.mms):
                ss = _m.prepare_enc([one], 1)
                hi, at = ss.get_results_one()
                ye = _m.get_start_yembs(1)
                sc = _m.get_score(at, hi, ye)
                prob = dy.softmax(sc)
                for z in range(self.expand):
                    # here, not the 'correct' order again, but since the first states are all the same ...
                    cur_hiddens[j][z].append(ss)
                    cur_probs[j][z].append(prob)
        self.hiddens = cur_hiddens
        # average scores
        return self._avg_probs(cur_probs)

    def feed(self, nexts, orders):
        # orders/nexts => list(batch) of list(expand) of int
        cur_hiddens = [[] for _ in self.bsize]
        cur_probs = [[] for _ in self.bsize]
        if orders is None:
            orders = [[i for i, cur_ne in enumerate(one_ne)] for one_ne in nexts]
        for j, one_ne, one_or in enumerate(zip(nexts, orders)):
            cur_hiddens[j].append([])
            cur_probs[j].append([])
            for z, cur_ne, cur_or in enumerate(zip(one_ne, one_or)):
                for i, _m in enumerate(self.mms):
                    ye = _m.get_embeddings_step(cur_ne, _m.embed_trg)
                    this_ss = self.hiddens[j][cur_or][i]
                    ss = _m.dec.feed_one(this_ss, ye)
                    hi, at = ss.get_results_one()
                    sc = _m.get_score(at, hi, ye)
                    cur_hiddens[j][-1].append(ss)
                    cur_probs[j][-1].append(dy.softmax(sc))
        self.hiddens = cur_hiddens
        # average scores
        return self._avg_probs(cur_probs)

# how to select the next actions
class SamplingStrategy(object):
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def start_hypo_num(self):
        return self.sample_size

    def width(self):
        return self.sample_size

    def _sample(self, probs):
        return np.argmax(np.random.multinomial(1, probs))

    def select_next(self, hypos, results):
        # return nexts, orders
        cands = []
        for i in range(len(hypos)):
            hs = hypos[i]
            candidates = []
            for j, h in enumerate(hs):
                # sample
                sc = results[i][j]
                nx = self._sample(sc)
                candidates.append((Hypo(last_action=nx, prev=h, score_one=np.log(sc[nx])), nx, j))
            cands.append(candidates)
        return cands

class BeamStrategy(object):
    def __init__(self, beam_size):
        self.beam_size = beam_size

    def start_hypo_num(self):
        return 1

    def width(self):
        return self.beam_size

    def select_next(self, hypos, results):
        # hypos: list(batch) of list(expand), results:(batch, expand, vocab)
        # RETURN candidates: (hypo, next, prev_index)
        cands = []
        for i, hs in enumerate(hypos):
            # generate the next beam --- first filter on this scores
            candidates = []
            for j, h in enumerate(hs):
                sc = results[i][j]
                top_ids = np.argpartition(sc, max(-len(sc),-self.beam_size))[-self.beam_size:]
                for one in top_ids:
                    candidates.append((Hypo(last_action=one, prev=h, score_one=np.log(sc[one])), one, j))
            # sort by candidates
            best_cands = sorted(candidates, key=lambda x: x[0].partial_score, reverse=True)[:self.beam_size]
            cands.append(best_cands)
        return cands
