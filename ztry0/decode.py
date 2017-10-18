# decoding with the model
import dynet as dy
import numpy as np
from . import utils
from . import data


# main decoding routine, the include_gold mode is only for debugging
def decode(diter, mms, target_dict, opts, outf):
    one_recorder = utils.OnceRecorder("DECODE")
    num_sents = len(diter)
    cur_sents = 0.
    bsize = diter.bsize()
    # decoding them all
    results = []
    prev_point = 0
    for xs, _1, _2, _3 in diter:
        if opts["verbose"] and (cur_sents - prev_point) >= (opts["report_freq"]*bsize):
            utils.printing("Decoding process: %.2f%%" % (cur_sents / num_sents * 100))
            prev_point = cur_sents
        cur_sents += len(xs)
        rs = search(xs, mms, opts, opts["decode_way"], opts["decode_batched"])
        results += rs
        one_recorder.record(xs, None, 0, 0)
    # restore from sorting by length
    results = diter.restore_sort_by_length(results)
    with utils.zfopen(outf, "w") as f:
        for r in results:
            best_seq = r[0].get_path("last_action")
            strs = data.Dict.i2w(target_dict, best_seq)
            f.write(" ".join(strs)+"\n")
    one_recorder.report()

# special decodig process
def decode_gold(diter, mms, target_dict, opts, outf, outfg):
    assert opts["debug"], "this should be debugging mode!!"
    one_recorder = utils.OnceRecorder("DECODE")
    num_sents = len(diter)
    cur_sents = cur_goldup = cur_correct = 0.
    bsize = diter.bsize()
    # decoding them all
    results = []
    results_gold = []
    prev_point = 0
    for xs, ys, _2, _3 in diter:
        if opts["verbose"] and (cur_sents - prev_point) >= (opts["report_freq"]*bsize):
            utils.printing("Decoding process: %.2f%%" % (cur_sents / num_sents * 100))
            prev_point = cur_sents
        cur_sents += len(xs)
        rs = search(xs, mms, opts, opts["decode_way"], opts["decode_batched"], ys)
        gs = build_ys(xs, ys, mms, opts)
        results += rs
        # compare them
        for _t, _g in zip(rs, gs):
            if Hypo.same_str(_t[0], _g[0]):
                cur_correct += 1
                results_gold.append(_t)
            elif _g[0].final_score > _t[0].final_score:
                cur_goldup += 1
                results_gold.append(_g)
            else:
                results_gold.append(_t)
        one_recorder.record(xs, None, 0, 0)
    utils.printing("Results of gold up: correct:%s, goldup:%s, all:%s" % (cur_correct, cur_goldup, cur_sents), func="score")
    # restore from sorting by length
    results = diter.restore_sort_by_length(results)
    results_gold = diter.restore_sort_by_length(results_gold)
    with utils.zfopen(outf, "w") as f:
        for r in results:
            best_seq = r[0].get_path("last_action")
            strs = data.Dict.i2w(target_dict, best_seq)
            f.write(" ".join(strs)+"\n")
    with utils.zfopen(outfg, "w") as f:
        for r in results_gold:
            best_seq = r[0].get_path("last_action")
            strs = data.Dict.i2w(target_dict, best_seq)
            f.write(" ".join(strs)+"\n")
    one_recorder.report()

# for debug printing at each step
def _debug_printing_step(s, xs, ys, hypos, results, cands, finished, src_d, trg_d, bsize, printing=True):
    def _tops(sc, size):
        return np.argpartition(sc, max(-len(sc),-size))[-size:]
    rets = []
    if ys is None:
        ys = [[0] for _ in xs]
    for x, y, hs, fis, rs, atts, cs in zip(xs, ys, hypos, finished, results["scores"], results["attws"], cands):
        r = {}
        r["src_ws"] = [src_d._getw(w[0]) for w in x]
        r["trg_ws"] = [trg_d._getw(w) for w in y]
        r["hypos"] = [[(one, trg_d._getw(one.last_action)) for one in h.get_path()] for h in hs]
        r["hypos_p"] = [(" ".join([trg_d._getw(one.last_action) for one in h.get_path()]), "%.3f"%h.partial_score, "%.3f"%h.final_score) for h in hs]
        r["finish"] = [[(one, trg_d._getw(one.last_action)) for one in h.get_path()] for h in fis]
        r["finish_p"] = [(" ".join([trg_d._getw(one.last_action) for one in h.get_path()]), "%.3f"%h.partial_score, "%.3f"%h.final_score) for h in fis]
        r["results"] = [[(trg_d._getw(one), "%.3f"%rs[i][one]) for one in _tops(rs[i], bsize)] for i in range(len(hs))]
        r["atts"] = [[(w, "%.3f"%a) for w, a in zip(r["src_ws"], att)] for att in atts]
        r["cands"] = [[prev_index, trg_d._getw(this_action), this_hypo] for this_hypo, this_action, prev_index in cs]
        rets.append(r)
    if printing:
        for ii, r in enumerate(rets):
            utils.printing("\nStep %s for %s" % (s, r["src_ws"]), func="debug")
            utils.printing("Original hypos are:", func="debug")
            for hi, zzz in enumerate(zip(r["hypos_p"], r["results"], r["atts"])):
                utils.printing("%s: %s" % (hi, zzz[0]), func="debug")
                utils.printing("!!SCORE -> %s" % zzz[1], func="debug")
                utils.printing("!!ATTS -> %s" % zzz[2], func="debug")
                utils.printing("=" * 20, func="debug")
            for zz in r["cands"]:
                utils.printing("!!CANDS-> %s" % ((" ".join([trg_d._getw(one.last_action) for one in zz[-1].get_path()]), zz),), func="debug")
            for zz in r["finish_p"]:
                utils.printing("!!FIN-> %s" % (zz,), func="debug")
            utils.printing("!!REF-> %s" % r["trg_ws"], func="debug")
    return rets

# build losses with fixed outputs: each has only one
def build_ys(xs, ys, models, opts):
    rets = []
    losses = [mm.fb2(xs, ys, False) for mm in models]
    normer = Normer() if (opts["normalize"] <= 0) else PolyNormer(opts["normalize"], opts["normalize_during_search"])
    for x, y, i in zip(xs, ys, range(len(xs))):
        hp = Hypo(normer=normer)
        for ss, tok in enumerate(y):
            lo = [ll[i][ss] for ll in losses]
            one_loss = -1*np.log(np.average([np.exp(-1*_x) for _x in lo]))
            hp = Hypo(last_action=tok, prev=hp, score_one=-1*one_loss, attws=None)    # todo: attws
        rets.append([hp])
    return rets

# the main searching routine
def search(xs, models, opts, strategy, batched, ys=None):
    # xs: list(batch) of list(sent) of int
    if type(models) not in [list, tuple]:
        models = [models]
    dy.renew_cg()
    for _mm in models:
        _mm.refresh(False, False)
    # prepare
    st = {"beam":BeamStrategy(opts["beam_size"]), "sample":SamplingStrategy(opts["sample_size"])}[strategy]
    if batched:
        pr = BatchedProcess(models, st.width(), opts["decode_batched_padding"])
    else:
        pr = NonBatchedProcess(models, st.width())
    normer = Normer() if (opts["normalize"] <= 0) else PolyNormer(opts["normalize"], opts["normalize_during_search"])
    hypos = [[Hypo(normer=normer) for _z in range(st.start_hypo_num())] for _ in range(len(xs))]
    finished = [[] for _ in range(len(xs))]
    # for max-steps
    nexts, orders = None, None
    eos_ind = models[0].target_dict.eos
    maxlen = min(opts["decode_len"], max([len(one)*opts["decode_ratio"] for one in xs]))
    for s in range(int(maxlen)):
        # feed one and get results
        if s == 0:
            results = pr.start(xs, st.start_hypo_num())
        else:
            results = pr.feed(nexts, orders)
        # get next steps
        cands = st.select_next(hypos, results)
        new_nexts, new_orders, new_hypos = [[] for _ in range(len(xs))], [[] for _ in range(len(xs))], [[] for _ in range(len(xs))]
        for i, one_cands in enumerate(cands):
            # todo(warn): decrease beam size here
            if len(finished[i]) >= st.width():
                continue
            one_cands = one_cands[:st.width()-len(finished[i])]
            for j, cc in enumerate(one_cands):
                this_hypo, this_action, prev_index = cc
                if this_action == eos_ind:
                    finished[i].append(this_hypo)
                else:
                    new_nexts[i].append(this_action)
                    new_orders[i].append(prev_index)
                    new_hypos[i].append(this_hypo)
        nexts, orders = new_nexts, new_orders
        hypos = new_hypos
        # print the status for debugging
        if opts["debug"] and opts["verbose"]:
            _debug_printing_step(s, xs, ys, hypos, results, [c[:st.width()-len(finished[i])] for i,c in enumerate(cands)],
                                    finished, models[0].source_dicts[0], models[0].target_dict, st.width())
        # test finished ?
        if all([len(fs) >= st.width() for fs in finished]):
            break
    # force eos
    if not all([len(i)==0 for i in nexts]):
        final_results = pr.feed(nexts, orders)
        for i, hs in enumerate(hypos):
            last_finishes = [Hypo(last_action=eos_ind, prev=h, score_one=final_results["scores"][i][j][eos_ind], attws=final_results["attws"][i][j])
                                for j, h in enumerate(hs)]
            last_finishes = Hypo.sort_hypos(last_finishes)
            if len(finished[i]) < st.width():
                finished[i] += last_finishes[:(st.width()-len(finished[i]))]
    # final check and ordering
    rets = [[f for f in Hypo.sort_hypos(fs)[:st.width()]] for fs in finished]
    # some usage printing
    if opts["debug"] and opts["verbose"]:
        utils.printing("Usage-info: %s" % pr.get_stat())
    return rets

# length normer
class Normer(object):
    def norm_score(self, s, l):
        return s

    def norm_score_partial(self, s, l):
        return s

class PolyNormer(Normer):
    def __init__(self, alpha, normalize_during_search):
        self.alpha = alpha
        self.normalize_during_search = normalize_during_search

    def norm_score(self, s, l):
        return s/pow(l, self.alpha)

    def norm_score_partial(self, s, l):
        if self.normalize_during_search:
            return self.norm_score(s, l)
        else:
            return s

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
        self._values = {"score_prob": lambda: np.exp(self.score_one)}

    def __repr__(self):
        return "%s/%s/%.3f/%.3f" % (self.length, self.last_action, self.score_one, self.score_acc)

    def __str__(self):
        return self.__repr__()

    @property
    def partial_score(self):
        return self.normer.norm_score_partial(self.score_acc, self.length)

    @property
    def final_score(self):
        return self.normer.norm_score(self.score_acc, self.length)

    def _get(self, which):
        if which is None:
            return self
        elif which in self._values:
            return self._values[which]()
        else:
            return getattr(self, which)

    def get_path(self, which=None):
        if self.prev is None:
            return []
        else:
            l = self.prev.get_path(which)
            l.append(self._get(which))
            return l

    @staticmethod
    def same_str(a, b):
        return a.length==b.length and all([x==y for x,y in zip(a.get_path("last_action"), b.get_path("last_action"))])

    @staticmethod
    def sort_hypos(ls):
        return sorted(ls, key=lambda x:x.final_score, reverse=True)

# how the batch of decoding states are represented
class Process(object):
    def __init__(self, mms, expand):
        self.mms = mms
        self.expand = expand
        self.bsize = -1
        # list(mms) of ss
        self.hiddens = None
        self.stat = {"l":0, "r":0}  # l=len, r=real

    @property
    def all_size(self):
        return self.expand*self.bsize

    def start(self, **argv):
        raise NotImplementedError()

    def feed(self, **argv):
        raise NotImplementedError()

    def _return(self, scores, attws):
        return {"scores":scores, "attws":attws}

    def get_stat(self):
        self.stat["b"] = self.bsize
        self.stat["w"] = self.expand
        self.stat["all"] = self.all_size * self.stat["l"]
        self.stat["perc"] = (self.stat["r"]+0.)/self.stat["all"]
        return self.stat

class BatchedProcess(Process):
    def __init__(self, mms, expand, padding):
        super(BatchedProcess, self).__init__(mms, expand)
        self.padding = padding
        self.prev_sizes = None
        self.cur_sizes = None

    def _fold_list(self, ss):
        # return list of list of scores (bs, expand, vocab)
        if self.padding:
            return np.asarray(ss).reshape((self.bsize, self.expand, -1))
        else:
            lines = sum(self.cur_sizes)
            sc = np.asarray(ss).reshape((lines, -1))
            ret = []
            base = 0
            for s in self.cur_sizes:
                ret.append(sc[base:base+s])
                base += s
            return ret

    def _flat_list(self, ll, sizes, check=False, check_rerange=False):
        utils.DEBUG_check(len(ll) == len(sizes))
        r = []
        base = 0
        flag = True
        for l, s in zip(ll, sizes):
            flag = flag and len(l) == s
            for i, one in enumerate(l):
                new_index = base + one
                flag = flag and one == i
                if check:
                    utils.DEBUG_check(one < s)
                r.append(new_index)
            base += s
        if check_rerange and flag:
            # if for each list, it is [0, 1, ...] up to sizes, then we don't need to rearange
            return None
        else:
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

    def start(self, xs, _):
        self.bsize = len(xs)
        # todo, bad-dependency
        self.stat["l"] += 1
        self.stat["r"] += self.all_size
        self.cur_sizes = [self.expand for _ in range(self.bsize)]
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
        # prev sizes
        self.prev_sizes = self.cur_sizes
        return self._return(scores=final_score, attws=avg_attws)

    def feed(self, nexts, orders):
        # todo, bad-dependency
        self.stat["l"] += 1
        self.stat["r"] += sum([len(i) for i in nexts])
        # nexts//orders => list(batch) of list(expand) of int
        self.cur_sizes = [len(one) for one in nexts]
        if self.padding:
            self._pad_list(nexts, orders)
        flat_nexts = self._flat_list(nexts, [0 for _ in nexts])
        flat_orders = self._flat_list(orders, self.prev_sizes, check=True, check_rerange=True)
        flat_att_orders = self._flat_list([[i for i in range(len(zz))] for zz in orders], self.prev_sizes,
                                          check=True, check_rerange=True)
        # start it
        cur_hiddens = []
        cur_probs = []
        cur_attws = []
        for i, _m in enumerate(self.mms):
            ye = _m.get_embeddings_step(flat_nexts, _m.embed_trg)
            if flat_orders is not None:
                self.hiddens[i].rerange_cache(flat_orders, flat_att_orders)
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
        # prev sizes
        if not self.padding:
            self.prev_sizes = self.cur_sizes
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
        # todo, bad-dependency
        self.stat["l"] += 1
        self.stat["r"] += self.all_size
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
        # todo, bad-dependency
        self.stat["l"] += 1
        self.stat["r"] += sum([len(i) for i in nexts])
        # orders/nexts => list(batch) of list(expand) of int
        cur_hiddens = [[] for _ in range(self.bsize)]
        cur_probs = [[] for _ in range(self.bsize)]
        cur_attws = [[] for _ in range(self.bsize)]
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
