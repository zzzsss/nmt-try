# decoding with the model
import dynet as dy
import numpy as np
import utils, data

def _check_order(l):
    return (len(l) == 1 and l[0] == 0) or (len(l)>1 and l[-1] == l[-2]+1)

def decode(diter, mms, target_dict, opts, outf):
    one_recorder = utils.OnceRecorder("DECODE")
    num_batches = len(diter)
    cur_batches = 0.
    # decoding them all
    results = []
    for xs, _1, _2, _3 in diter:
        if opts["verbose"] and cur_batches % opts["report_freq"] == 0:
            utils.printing("Decoding process: %.2f%%" % (cur_batches/num_batches*100))
        cur_batches += 1
        rs = search(xs, mms, opts, opts["decode_way"], opts["decode_batched"])
        results += rs
        one_recorder.record(xs, None, 0, 0)
    # restore from sorting by length
    diter.restore_sort_by_length(results)
    with utils.zfopen(outf, "w") as f:
        for r in results:
            best_seq = [one.last_action for one in r[0]]
            strs = data.Dict.i2w(target_dict, best_seq)
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
            results = pr.start(xs, st.start_hypo_num())
        else:
            results = pr.feed(nexts, orders)
        # get next steps
        cands = st.select_next(hypos, results)
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
        hypos = new_hypos
        # test finished ?
        if all([len(fs) >= st.width() for fs in finished]):
            break
    # force eos
    final_results = pr.feed(nexts, orders)
    for i, hs in enumerate(hypos):
        for j, h in enumerate(hs):
            hf = Hypo(last_action=eos_ind, prev=h, score_one=final_results["scores"][i][j][eos_ind], attws=final_results["attws"][i][j])
            finished[i].append(hf)
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
    def __init__(self, last_action=None, prev=None, score_one=0., normer=None, attws=None):
        self.last_action = last_action    # None for the start symbol
        self.prev = prev
        self.score_one = score_one
        self.score_acc = prev.score_acc+score_one if prev else 0.
        self.length = prev.length+1 if prev is not None else 0
        self.normer = None
        if normer is not None:
            self.normer = normer
        elif self.prev is not None:
            self.normer = self.prev.normer
        else:
            self.normer = Hypo.DefaultNormer()
        self.attws = None
        if attws is not None:
            self.attws = [float(x) for x in attws]    # attention weights

    @property
    def partial_score(self):
        return self.normer.norm_score(self.score_acc, self.length)

    @property
    def final_score(self):
        return self.normer.norm_score(self.score_acc, self.length)

    def _get(self, which):
        if which is None:
            return self
        else:
            return getattr(self, which)

    def get_path(self, which=None):
        if self.prev is None:
            return []
        else:
            l = self.prev.get_path(which)
            l.append(self._get(which))
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

    def _return(self, scores, attws):
        return {"scores":scores, "attws":attws}

class BatchedProcess(Process):
    def __init__(self, mms, expand):
        super(BatchedProcess, self).__init__(mms, expand)

    def _fold_list(self, ss):
        # return list of list of scores (bs, expand, vocab)
        return np.asarray(ss).reshape((self.bsize, self.expand, -1))

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

    def _avg(self, l):
        if len(l)==1:
            r = l[0]
        else:
            r = dy.average(l)
        return self._fold_list(r.value())

    def start(self, xs, start_expand):
        self.bsize = len(xs)
        cur_hiddens = []
        cur_probs = []
        cur_attws = []
        for _m in self.mms:
            ss = _m.prepare_enc(xs, self.expand)
            hi, at, atw = ss.get_results_one()
            ye = _m.get_start_yembs(self.all_size)
            sc = _m.get_score(at, hi, ye)
            cur_hiddens.append(ss)
            cur_probs.append(dy.softmax(sc))
            cur_attws.append(atw)
        self.hiddens = cur_hiddens
        # average scores
        final_score = self._avg(cur_probs)
        # average attentions, but currently we don't actually use it
        avg_attws = self._avg(cur_attws)
        return self._return(scores=final_score, attws=avg_attws)

    def feed(self, nexts, orders):
        # orders/nexts => list(batch) of list(expand) of int
        self._pad_list(nexts, orders)
        flat_nexts = self._flat_list(nexts, False)
        flat_orders = self._flat_list(orders, True) if orders is not None else None
        cur_hiddens = []
        cur_probs = []
        cur_attws = []
        for i, _m in enumerate(self.mms):
            ye = _m.get_embeddings_step(flat_nexts, _m.embed_trg)
            if flat_orders is not None:     # no change if None (for eg., when sampling)
                self.hiddens[i].shuffle(flat_orders)
            ss = _m.dec.feed_one(self.hiddens[i], ye)
            hi, at, atw = ss.get_results_one()
            sc = _m.get_score(at, hi, ye)
            cur_hiddens.append(ss)
            cur_probs.append(dy.softmax(sc))
            cur_attws.append(atw)
        self.hiddens = cur_hiddens
        # average scores
        final_score = self._avg(cur_probs)
        # average attentions, but currently we don't actually use it
        avg_attws = self._avg(cur_attws)
        return self._return(scores=final_score, attws=avg_attws)

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

    def start(self, xs, start_expand):
        self.bsize = len(xs)
        cur_hiddens = [[[] for _ in range(start_expand)] for _j in xs]
        cur_probs = [[[] for _ in range(start_expand)] for _j in xs]
        cur_attws = [[[] for _ in range(start_expand)] for _j in xs]
        for j, one in enumerate(xs):
            for i, _m in enumerate(self.mms):
                ss = _m.prepare_enc([one], 1)
                hi, at, atw = ss.get_results_one()
                ye = _m.get_start_yembs(1)
                sc = _m.get_score(at, hi, ye)
                prob = dy.softmax(sc)
                for z in range(start_expand):
                    # here, not the 'correct' order again, but since the first states are all the same ...
                    cur_hiddens[j][z].append(ss)
                    cur_probs[j][z].append(prob)
                    cur_attws[j][z].append(atw)
        self.hiddens = cur_hiddens
        # average scores
        return self._return(scores=self._avg_probs(cur_probs), attws=self._avg_probs(cur_attws))

    def feed(self, nexts, orders):
        # orders/nexts => list(batch) of list(expand) of int
        cur_hiddens = [[] for _ in range(self.bsize)]
        cur_probs = [[] for _ in range(self.bsize)]
        cur_attws = [[] for _ in range(self.bsize)]
        if orders is None:
            orders = [[i for i, cur_ne in enumerate(one_ne)] for one_ne in nexts]
        for j in range(len(nexts)):
            one_ne, one_or = nexts[j], orders[j]
            for z in range(len(one_ne)):
                cur_hiddens[j].append([])
                cur_probs[j].append([])
                cur_attws[j].append([])
                cur_ne, cur_or = one_ne[z], one_or[z]
                for i, _m in enumerate(self.mms):
                    ye = _m.get_embeddings_step(cur_ne, _m.embed_trg)
                    this_ss = self.hiddens[j][cur_or][i]
                    ss = _m.dec.feed_one(this_ss, ye)
                    hi, at, atw = ss.get_results_one()
                    sc = _m.get_score(at, hi, ye)
                    cur_hiddens[j][-1].append(ss)
                    cur_probs[j][-1].append(dy.softmax(sc))
                    cur_attws[j][-1].append(atw)
        self.hiddens = cur_hiddens
        # average scores
        return self._return(scores=self._avg_probs(cur_probs), attws=self._avg_probs(cur_attws))

# how to select the next actions
class SamplingStrategy(object):
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def start_hypo_num(self):   # this must be < width
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
                sc = results["scores"][i][j]
                attws = results["attws"][i][j]
                nx = self._sample(sc)
                candidates.append((Hypo(last_action=nx, prev=h, score_one=np.log(sc[nx]), attws=attws), nx, j))
            cands.append(candidates)
        return cands

class BeamStrategy(object):
    def __init__(self, beam_size, hypo_expand_size=-1):
        self.beam_size = beam_size
        self.hypo_expand_size = hypo_expand_size if hypo_expand_size>0 else beam_size

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
                sc = results["scores"][i][j]
                attws = results["attws"][i][j]
                top_ids = np.argpartition(sc, max(-len(sc),-self.hypo_expand_size))[-self.hypo_expand_size:]
                for one in top_ids:
                    next_one = int(one)
                    candidates.append((Hypo(last_action=next_one, prev=h, score_one=np.log(sc[next_one]), attws=attws), next_one, j))
            # sort by candidates
            best_cands = sorted(candidates, key=lambda x: x[0].partial_score, reverse=True)[:self.beam_size]
            cands.append(best_cands)
        return cands
