# focusing on speed

from zl.model import Model
from zl.trainer import Trainer
from zl import utils, data
from . import mt_search, mt_eval
from .mt_length import MTLengther, LinearGaussain
from collections import defaultdict
import numpy as np

from . import mt_layers as layers
from .mt_length import get_normer
from .mt_outputter import Outputter

# herlpers
# data helpers #
def prepare_data(ys, dict, fix_len=0):
    # input: list of list of index (bsize, step),
    # output: padded input (step, bsize), masks (1 for real words, 0 for paddings, None for all 1)
    bsize, steps = len(ys), max([len(i) for i in ys])
    if fix_len > 0:
        steps = fix_len
    y = [[] for _ in range(steps)]
    ym = [None for _ in range(steps)]
    lens = [len(i) for i in ys]
    eos, padding = dict.eos, dict.pad
    for s in range(steps):
        _one_ym = []
        _need_ym = False
        for b in range(bsize):
            if s<lens[b]:
                y[s].append(ys[b][s])
                _one_ym.append(1.)
            else:
                y[s].append(padding)
                _one_ym.append(0.)
                _need_ym = True
        if _need_ym:
            ym[s] = _one_ym
    # check last step
    for b in range(bsize):
        if y[-1][b] not in [eos, padding]:
            y[-1][b] = eos
    return y, ym

def prepare_y_step(ys, i):
    _need_mask = False
    ystep = []
    _mask = []
    for _y in ys:
        if i<len(_y):
            ystep.append(_y[i])
            _mask.append(1.)
        else:
            ystep.append(0)
            _mask.append(0)
            _need_mask = True
    if _need_mask:
        mask_expr = layers.BK.inputVector(_mask)
        mask_expr = layers.BK.reshape(mask_expr, (1, ), len(ys))
    else:
        mask_expr = None
    return ystep, mask_expr

# An typical example of a model, fixed architecture
# single s2s: one input(no factors), one output
# !! stateless, states & caches are managed by the Scorer
class s2sModel(Model):
    def __init__(self, opts, source_dict, target_dict, length_info):
        super(s2sModel, self).__init__()
        self.opts = opts
        self.source_dict = source_dict
        self.target_dict = target_dict
        # build the layers
        # embeddings
        self.embed_src = layers.Embedding(self.model, len(source_dict), opts["dim_word"], dropout_wordceil=source_dict.get_wordceil())
        self.embed_trg = layers.Embedding(self.model, len(target_dict), opts["dim_word"], dropout_wordceil=target_dict.get_wordceil())
        # enc-dec
        self.enc = layers.Encoder(self.model, opts["dim_word"], opts["hidden_enc"], opts["enc_depth"], opts["rnn_type"])
        self.dec = layers.NematusDecoder(self.model, opts["dim_word"], opts["hidden_dec"], opts["dec_depth"], 2*opts["hidden_enc"],
                    opts["hidden_att"], opts["att_type"], opts["rnn_type"], opts["summ_type"])
        # outputs
        self.out0 = layers.Linear(self.model, 2*opts["hidden_enc"]+opts["hidden_dec"]+opts["dim_word"], opts["hidden_out"])
        self.out1 = layers.Linear(self.model, opts["hidden_out"], len(target_dict), act="linear")
        self.model_softmax = opts["model_softmax"]
        self.show_loss = opts["show_loss"]
        #
        # computation values
        # What is in the cache: S,V,summ/ ctx,att,/ out_s,results
        self.names_bv = {"hid"}
        self.names_bi = {"S", "V"}
        self.names_ig = {"ctx", "att", "out_s", "results", "summ"}
        # length
        self.scaler = MTLengther.get_scaler_f(opts["train_scale_way"], opts["train_scale"])  # for training
        self.lg = LinearGaussain(self.model, 2*opts["hidden_enc"], opts["train_len_xadd"], opts["train_len_xback"], length_info)
        utils.zlog("End of creating Model.")
        # !! advanced options (enabled by MTTrainer)
        self.is_fitting_length = False      # whether adding length loss for training
        self.len_lambda = opts["train_len_lambda"]

    def rerange(self, c, bv_orders, bi_orders):
        new_c = {}
        for names, orders in ((self.names_bv, bv_orders), (self.names_bi, bi_orders)):
            if orders is not None:
                for n in names:
                    new_c[n] = layers.BK.rearrange_cache(c[n], orders)
            else:
                for n in names:
                    new_c[n] = c[n]
        return new_c

    def refresh(self, training):
        def _gd(drop):  # get dropout
            return drop if training else 0.
        opts = self.opts
        self.embed_src.refresh({"hdrop":_gd(opts["drop_embedding"]), "idrop":_gd(opts["idrop_embedding"])})
        self.embed_trg.refresh({"hdrop":_gd(opts["drop_embedding"]), "idrop":_gd(opts["idrop_embedding"])})
        self.enc.refresh({"idrop":_gd(opts["idrop_enc"]), "gdrop":_gd(opts["gdrop_enc"])})
        self.dec.refresh({"idrop":_gd(opts["idrop_dec"]), "gdrop":_gd(opts["gdrop_dec"]), "hdrop":_gd(opts["drop_hidden"])})
        self.out0.refresh({"hdrop":_gd(opts["drop_hidden"])})
        self.out1.refresh({})
        self.lg.refresh({"idrop":_gd(opts["drop_hidden"])})

    def update_schedule(self, uidx):
        # todo, change mode while training (before #num updates)
        # fitting len
        if not self.is_fitting_length and uidx>=self.opts["train_len_uidx"]:
            self.is_fitting_length = True
            utils.zlog("(Advanced-fitting) Model is starting to fit length.")

    # helper routines #
    def get_embeddings_step(self, tokens, embed):
        # tokens: list of int or one int, embed: Embedding => one expression (batched)
        return embed(tokens)

    def get_start_yembs(self, bsize):
        return self.get_embeddings_step([self.target_dict.bos for _ in range(bsize)], self.embed_trg)

    def get_scores(self, at, hi, ye):
        real_hi = hi[-1]["H"]
        output_concat = layers.BK.concatenate([at, real_hi, ye])
        output_hidden = self.out0(output_concat)
        output_score = self.out1(output_hidden)
        return output_score, output_hidden

    def encode(self, xx, xm):
        # -- encode xs, return list of encoding vectors
        # xx, xm = self.prepare_data(xs) # prepare at the outside
        x_embed = [self.get_embeddings_step(s, self.embed_src) for s in xx]
        x_encodes = self.enc(x_embed, xm)
        return x_encodes

    def decode_start(self, x_encodes):
        # start the first step of decoding
        return self.dec.start_one(x_encodes)

    def decode_step(self, x_encodes, inputs, caches):
        # feed one step
        return self.dec.feed_one(x_encodes, inputs, caches)

    # main routines #
    def start(self, xs, repeat=1, softmax=True):
        # encode
        bsize = len(xs)
        xx, xm = prepare_data(xs, self.source_dict)
        x_encodes = self.encode(xx, xm)
        x_encodes = [layers.BK.batch_repeat(one, repeat) for one in x_encodes]
        # init decode
        cache = self.decode_start(x_encodes)
        start_embeds = self.get_start_yembs(bsize)
        output_score, output_hidden = self.get_scores(cache["ctx"], cache["hid"], start_embeds)
        if softmax and self.model_softmax:
            results = layers.BK.softmax(output_score)
        else:
            results = output_score
        # return
        cache["out_s"] = output_score
        cache["results"] = results
        return cache

    def step(self, prev_val, inputs, softmax=True):
        x_encodes, hiddens = None, prev_val["hid"]
        next_embeds = self.get_embeddings_step(inputs, self.embed_trg)
        cache = self.decode_step(x_encodes, next_embeds, prev_val)
        output_score, output_hidden = self.get_scores(cache["ctx"], cache["hid"], next_embeds)
        if softmax and self.model_softmax:
            results = layers.BK.softmax(output_score)
        else:
            results = output_score
        # return
        cache["out_s"] = output_score
        cache["results"] = results
        return cache

    # for the results
    def explain_result(self, x):
        if self.model_softmax:
            return np.log(x)
        else:
            return x

    # training
    def fb(self, insts, training):
        xs = [i[0] for i in insts]
        ys = [i[1] for i in insts]
        Model.new_graph()
        self.refresh(training)
        bsize = len(xs)
        # opens = [State(sg=SearchGraph()) for _ in range(bsize)]     # with only one init state
        xlens = [len(_x) for _x in xs]
        ylens = [len(_y) for _y in ys]
        cur_maxlen = max(ylens)
        losses = []
        caches = []
        yprev = None
        for i in range(cur_maxlen):
            # forward
            ystep, mask_expr = prepare_y_step(ys, i)
            if i==0:
                cc = self.start(xs, softmax=False)
                if self.is_fitting_length:
                    pred_lens = self.lg.calculate(cc["summ"], xlens)
                    loss_len = self.lg.ll_loss(pred_lens, ylens)
            else:
                cc = self.step(caches[-1], yprev, softmax=False)
            caches.append(cc)
            yprev = ystep
            # build loss
            scores_exprs = cc["out_s"]
            loss = _mle_loss_step(scores_exprs, ystep, mask_expr)
            if self.scaler is not None:
                len_scales = [self.scaler(len(_y)) for _y in ys]
                np.asarray(len_scales).reshape((1, -1))
                len_scalue_e = layers.BK.inputTensor(len_scales, True)
                loss = loss * len_scalue_e
            losses.append(loss)
            # prepare next steps: only following gold
            # new_opens = [State(prev=ss, action=Action(yy, 0.)) for ss, yy in zip(opens, ystep)]
            # opens = new_opens
        # -- final
        loss0 = layers.BK.esum(losses)
        loss_y = layers.BK.sum_batches(loss0) / bsize
        if self.is_fitting_length:
            loss = loss_y + loss_len * self.len_lambda      # todo(warn): must be there
        else:
            loss = loss_y
        if training:
            layers.BK.forward(loss)
            layers.BK.backward(loss)
        # return value?
        if self.show_loss:
            lossy_val = layers.BK.get_value_sca(loss_y)
            loss_len_val = -1.
            if self.is_fitting_length:
                loss_len_val = layers.BK.get_value_sca(loss_len)
            return {"y": lossy_val*bsize, "len": loss_len_val}
        else:
            return {}

    def predict_length(self, insts, cc=None):
        # todo(warn): already inited graph
        # return real lengths
        xs = [i[0] for i in insts]
        xlens = [len(_x) for _x in xs]
        if cc is None:
            cc = self.start(xs, softmax=False)
        pred_lens = self.lg.calculate(cc["summ"], xlens)
        ret = np.asarray(layers.BK.get_value_vec(pred_lens))
        ret = LinearGaussain.back_len(ret)
        return ret

def _mle_loss_step(scores_exprs, ystep, mask_expr):
    one_loss = layers.BK.pickneglogsoftmax_batch(scores_exprs, ystep)
    if mask_expr is not None:
        one_loss = one_loss * mask_expr
    return one_loss

ValidResult = list

class OnceRecorder(object):
    def __init__(self, name):
        self.name = name
        self.loss = defaultdict(float)
        self.sents = 1e-6
        self.words = 1e-6
        self.updates = 0
        self.timer = utils.Timer("")

    def record(self, insts, loss, update):
        for k in loss:
            self.loss[k] += loss[k]
        self.sents += len(insts)
        self.words += sum([len(x[0]) for x in insts])     # for src
        self.updates += update

    def reset(self):
        self.loss = self.loss = defaultdict(float)
        self.sents = 1e-6
        self.words = 1e-6
        self.updates = 0
        self.timer = utils.Timer("")

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
            loss = self._mm.fb(insts, False)
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
        return OnceRecorder(name)

    def _fb_once(self, insts):
        return self._mm.fb(insts, True)

def mt_decode(decode_way, test_iter, mms, target_dict, opts, outf):
    cur_searcher = {"greedy":mt_search.search_greedy, "beam":mt_search.search_beam}[decode_way]
    one_recorder = OnceRecorder("DECODE")
    num_sents = len(test_iter)
    cur_sents = 0.
    # decoding them all
    results = []
    prev_point = 0
    # init normer
    for i, _m in enumerate(mms):
        _lg_params = _m.lg.obtain_params()
        utils.zlog("Model[%s] is with lg as %s." % (i, _lg_params,))
    _sigma = np.average([_m.lg.get_real_sigma() for _m in mms], axis=0)
    normer = get_normer(opts["normalize_way"], opts["normalize_alpha"], _sigma)
    for insts in test_iter.arrange_batches():
        if opts["verbose"] and (cur_sents - prev_point) >= (opts["report_freq"]*test_iter.bsize()):
            utils.zlog("Decoding process: %.2f%%" % (cur_sents / num_sents * 100))
            prev_point = cur_sents
        cur_sents += len(insts)
        mt_search.search_init()
        # return list(batch) of list(beam) of states
        rs = cur_searcher(mms, insts, target_dict, opts, normer)
        results += rs
        one_recorder.record(insts, {}, 0)
    one_recorder.report()
    # restore from sorting by length
    results = test_iter.restore_order(results)
    # output
    ot = Outputter(opts)
    # -- write one best
    with utils.zopen(outf, "w") as f:
        for r in results:
            f.write(ot.format(r, target_dict, False, False))
    # -- write k-best
    output_kbest, output_score = opts["decode_output_kbest"], opts["decode_output_score"]
    if output_kbest:
        # todo(warn): specified file name
        with utils.zopen(outf+".nbest", "w") as f:
            for r in results:
                f.write(ot.format(r, target_dict, True, output_score))
