from zl.trainer import Trainer
from zl.search2 import StateStater
from zl import data, utils
from zl.model import Model
from .mt_length import get_normer
from .mt_outputter import Outputter
from . import mt_search, mt_eval
from collections import defaultdict, Iterable
import numpy as np

ValidResult = list

class OnceRecorder(object):
    def __init__(self, name, mm=None):
        self.name = name
        self.loss = defaultdict(float)
        self.sents = 1e-6
        self.words = 1e-6
        self.updates = 0
        self.timer = utils.Timer("")
        self._mm = mm

    def record(self, insts, loss, update):
        for k in loss:
            self.loss[k] += loss[k]
        self.sents += len(insts)
        self.words += sum([len(x[0]) for x in insts])     # for src
        self.updates += update

    def reset(self):
        self.loss = self.loss = defaultdict(float)
        self.sents = 1e-5
        self.words = 1e-5
        self.updates = 0
        self.timer = utils.Timer("")
        #
        if self._mm is not None:
            self._mm.stat_clear()

    # const, only reporting, could be called many times
    def state(self):
        one_time = self.timer.get_time()
        loss_per_sentence = "_".join(["%s:%.3f"%(k, self.loss[k]/self.sents) for k in sorted(self.loss.keys())])
        loss_per_word = "_".join(["%s:%.3f"%(k, self.loss[k]/self.words) for k in sorted(self.loss.keys())])
        sent_per_second = float(self.sents) / one_time
        word_per_second = float(self.words) / one_time
        return ("Recoder <%s>, %.3f(time)/%s(updates)/%.1f(sents)/%.1f(words)/%s(sl-loss)/%s(w-loss)/%.3f(s-sec)/%.3f(w-sec)" % (self.name, one_time, self.updates, self.sents, self.words, loss_per_sentence, loss_per_word, sent_per_second, word_per_second))

    def report(self, s=""):
        utils.zlog(s+self.state(), func="info")
        if self._mm is not None:
            self._mm.stat_report()

class MTTrainer(Trainer):
    def __init__(self, opts, model):
        super(MTTrainer, self).__init__(opts, model)

    def _validate_len(self, dev_iter):
        # sqrt error
        count = 0
        loss = 0.
        with utils.Timer(tag="VALID-LEN", print_date=True) as et:
            utils.zlog("With lg as %s." % (self._mm.lg.obtain_params(),))
            for insts in dev_iter.arrange_batches():
                ys = [i[1] for i in insts]
                ylens = np.asarray([len(_y) for _y in ys])
                count += len(ys)
                Model.new_graph()
                self._mm.refresh(False)
                preds = self._mm.predict_length(insts)
                loss += np.sum((preds - ylens) ** 2)
        return - loss / count

    def _validate_ll(self, dev_iter):
        # log likelihood
        one_recorder = self._get_recorder("VALID-LL")
        for insts in dev_iter.arrange_batches():
            loss = self._mm.fb(insts, False, run_name="std2")   # only run fb-mle
            one_recorder.record(insts, loss, 0)
        one_recorder.report()
        # todo(warn) "y" as the key
        return -1 * (one_recorder.loss["y"] / one_recorder.words)

    def _validate_bleu(self, dev_iter):
        # bleu score
        # todo(warn): force greedy validating here
        mt_decode("greedy", dev_iter, [self._mm], self._mm.target_dict, self.opts, self.opts["dev_output"])
        # no restore specifies for the dev set
        s = mt_eval.evaluate(self.opts["dev_output"], self.opts["dev"][1], self.opts["eval_metric"], True)
        return s

    def _validate_them(self, dev_iter, metrics):
        validators = {"ll": self._validate_ll, "bleu": self._validate_bleu, "len": self._validate_len}
        r = []
        for m in metrics:
            s = validators[m](dev_iter)
            r.append(float("%.3f" % s))
        return ValidResult(r)

    def _get_recorder(self, name):
        return OnceRecorder(name, self._mm)

    def _fb_once(self, insts):
        return self._mm.fb(insts, True)

def mt_decode(decode_way, test_iter, mms, target_dict, opts, outf, gold_iter=None):
    reranking = (gold_iter is not None)
    if reranking:
        cur_searcher = mt_search.search_rerank
    else:
        cur_searcher = {"greedy":mt_search.search_greedy, "beam":mt_search.search_beam,
                        "sample":mt_search.search_sample, "branch":mt_search.search_branch}[decode_way]
    one_recorder = OnceRecorder("DECODE")
    num_sents = len(test_iter)
    cur_sents = 0.
    sstater = StateStater()
    # decoding them all
    results = []
    prev_point = 0
    # init normer
    for i, _m in enumerate(mms):
        _lg_params = _m.lg.obtain_params()
        utils.zlog("Model[%s] is with lg as %s." % (i, _lg_params,))
    _sigma = np.average([_m.lg.get_real_sigma() for _m in mms], axis=0)
    normer = get_normer(opts["normalize_way"], opts["normalize_alpha"], _sigma)
    # todo: ugly code here
    if isinstance(test_iter, Iterable):
        rs = cur_searcher(mms, test_iter, target_dict, opts, normer, sstater)
        results += rs
    else:
        if reranking:
            for one_tests, one_golds in zip(test_iter.arrange_batches(), gold_iter.arrange_batches()):
                if opts["verbose"] and (cur_sents - prev_point) >= (opts["report_freq"]*test_iter.bsize()):
                    utils.zlog("Reranking process: %.2f%%" % (cur_sents / num_sents * 100))
                    prev_point = cur_sents
                cur_sents += len(one_tests)
                mt_search.search_init()
                rs = cur_searcher(mms, [one_tests, one_golds], target_dict, opts, normer, sstater)
                results += rs
                one_recorder.record(one_tests, {}, 0)
        else:
            for insts in test_iter.arrange_batches():
                if opts["verbose"] and (cur_sents - prev_point) >= (opts["report_freq"]*test_iter.bsize()):
                    utils.zlog("Decoding process: %.2f%%" % (cur_sents / num_sents * 100))
                    prev_point = cur_sents
                cur_sents += len(insts)
                mt_search.search_init()
                # return list(batch) of list(beam) of states
                rs = cur_searcher(mms, insts, target_dict, opts, normer, sstater)
                results += rs
                one_recorder.record(insts, {}, 0)
        one_recorder.report()
        # restore from sorting by length
        results = test_iter.restore_order(results)
    # output
    sstater.report()
    # -- write one best
    output_r2l = opts["decode_output_r2l"]
    with utils.zopen(outf, "w") as f:
        ot = Outputter(opts)
        for r in results:
            f.write(ot.format(r, target_dict, False, False, output_r2l))
        ot.report()
    # -- write k-best
    output_kbest = opts["decode_output_kbest"]
    if output_kbest:
        # todo(warn): specified file names
        with utils.zopen(outf+".nbest", "w") as f:
            ot = Outputter(opts)
            for r in results:
                f.write(ot.format(r, target_dict, True, False, output_r2l))
            ot.report()
        with utils.zopen(outf+".nbests", "w") as f:
            ot = Outputter(opts)
            for r in results:
                f.write(ot.format(r, target_dict, True, True, output_r2l))
            ot.report()
        with utils.zopen(outf+".nbestg", "w") as f:
            for i, r in enumerate(results):
                f.write("# znumber%s\n%s\n" % (i, r[0].sg.show_graph(target_dict, False)))
        # todo: maybe dump sg
    return results

# fb_beamer's stater (like StateStater, but for training)
class FbeamStater(object):
    def __init__(self):
        pass


