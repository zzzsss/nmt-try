# focusing on speed

from zl.model import Model
from zl.trainer import Trainer
from zl import layers, utils, data
import numpy
from zl.layers import BK
from collections import Iterable
from . import mt_search, mt_eval
from zl.search import State, SearchGraph, Action
from .mt_length import MTLengther

# herlpers
# data helpers #
def prepare_data(ys, dict, fix_len=0):
    # input: list of list of index (bsize, step),
    # output: padded input (step, bsize), masks (1 for real words, 0 for paddings)
    bsize, steps = len(ys), max([len(i) for i in ys])
    if fix_len > 0:
        steps = fix_len
    y = [[] for _ in range(steps)]
    lens = [len(i) for i in ys]
    eos, padding = dict.eos, dict.pad
    for s in range(steps):
        for b in range(bsize):
            y[s].append(ys[b][s] if s<lens[b] else padding)
    # check last step
    for b in range(bsize):
        if y[-1][b] not in [eos, padding]:
            y[-1][b] = eos
    return y, [[(1. if len(one)>s else 0.) for one in ys] for s in range(steps)]

def prepare_y_step(ys, i):
    _mask = [1. if i<len(_y) else 0. for _y in ys]
    ystep = [_y[i] if i<len(_y) else 0 for _y in ys]
    mask_expr = layers.BK.inputVector(_mask)
    mask_expr = layers.BK.reshape(mask_expr, (1, ), len(ys))
    return ystep, mask_expr

# An typical example of a model, fixed architecture
# single s2s: one input(no factors), one output
# !! stateless, states & caches are managed by the Scorer
class s2sModel(Model):
    def __init__(self, opts, source_dict, target_dict):
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
        self.dec = layers.NematusDecoder(self.model, opts["dim_word"], opts["hidden_dec"], opts["dec_depth"],
                    opts["rnn_type"], opts["summ_type"], opts["att_type"], opts["hidden_att"], opts["dim_cov"], 2*opts["hidden_enc"])
        # outputs
        self.out0 = layers.Affine(self.model, 2*opts["hidden_enc"]+opts["hidden_dec"]+opts["dim_word"], opts["hidden_out"])
        self.out1 = layers.AffineNodrop(self.model, opts["hidden_out"], len(target_dict), act="linear")
        #
        # computation values
        # What is in the cache: S,V,summ/ cov,ctx,att,/ out_s,results
        self.names_bv = {"cov", "ctx", "hid"}
        self.names_bi = {"S", "V", "summ"}
        self.names_ig = {"att", "out_s", "results"}
        # length
        self.scaler = MTLengther.get_scaler_f(opts["train_scale_way"], opts["train_scale"])  # for training
        self.normer = None
        utils.zlog("End of creating Model.")

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

    # helper routines #
    def get_embeddings_step(self, tokens, embed):
        # tokens: list of int or one int, embed: Embedding => one expression (batched)
        return embed(tokens)

    def get_start_yembs(self, bsize):
        return self.get_embeddings_step([self.target_dict.bos for _ in range(bsize)], self.embed_trg)

    def get_scores(self, at, hi, ye):
        real_hi = hi[-1]["H"]
        output_concat = BK.concatenate([at, real_hi, ye])
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

    def decode_step(self, x_encodes, inputs, hiddens, caches):
        # feed one step
        return self.dec.feed_one(x_encodes, inputs, hiddens, caches)

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
        if softmax:
            probs = layers.BK.softmax(output_score)
        else:
            probs = output_score
        # return
        cache["out_s"] = output_score
        cache["results"] = probs
        return cache

    def step(self, prev_val, inputs, softmax=True):
        x_encodes, hiddens = None, prev_val["hid"]
        next_embeds = self.get_embeddings_step(inputs, self.embed_trg)
        cache = self.decode_step(x_encodes, next_embeds, hiddens, prev_val)
        output_score, output_hidden = self.get_scores(cache["ctx"], cache["hid"], next_embeds)
        if softmax:
            probs = layers.BK.softmax(output_score)
        else:
            probs = output_score
        # return
        cache["out_s"] = output_score
        cache["results"] = probs
        return cache

    # training
    def fb(self, insts, training):
        xs = [i[0] for i in insts]
        ys = [i[1] for i in insts]
        Model.new_graph()
        self.refresh(training)
        bsize = len(xs)
        # opens = [State(sg=SearchGraph()) for _ in range(bsize)]     # with only one init state
        cur_maxlen = max([len(_y) for _y in ys])
        losses = []
        caches = []
        yprev = None
        for i in range(cur_maxlen):
            # forward
            ystep, mask_expr = prepare_y_step(ys, i)
            if i==0:
                cc = self.start(xs, softmax=False)
            else:
                cc = self.step(caches[-1], yprev, softmax=False)
            caches.append(cc)
            yprev = ystep
            # build loss
            scores_exprs = cc["out_s"]
            loss = _mle_loss_step(scores_exprs, ystep, mask_expr)
            len_scales = layers.BK.inputVector([self.scaler(len(_y)) for _y in ys])
            len_scales = layers.BK.reshape(len_scales, (1, ), len(ys))
            loss = loss * len_scales
            losses.append(loss)
            # prepare next steps: only following gold
            # new_opens = [State(prev=ss, action=Action(yy, 0.)) for ss, yy in zip(opens, ystep)]
            # opens = new_opens
        # -- final
        loss = layers.BK.esum(losses)
        loss = layers.BK.sum_batches(loss) / bsize
        if training:
            # layers.BK.forward(loss)
            layers.BK.backward(loss)
        # return value?
        loss_val = layers.BK.get_value_sca(loss)
        return loss_val*bsize

def _mle_loss_step(scores_exprs, ystep, mask_expr):
    one_loss = layers.BK.pickneglogsoftmax_batch(scores_exprs, ystep)
    one_loss = one_loss * mask_expr
    return one_loss

ValidResult = list

class OnceRecorder(object):
    def __init__(self, name):
        self.name = name
        self.loss = 0.
        self.sents = 1e-6
        self.words = 1e-6
        self.updates = 0
        self.timer = utils.Timer("")

    def record(self, insts, loss, update):
        self.loss += loss
        self.sents += len(insts)
        self.words += sum([len(x[0]) for x in insts])     # for src
        self.updates += update

    def reset(self):
        self.loss = 0.
        self.sents = 1e-6
        self.words = 1e-6
        self.updates = 0
        self.timer = utils.Timer("")

    # const, only reporting, could be called many times
    def state(self):
        one_time = self.timer.get_time()
        loss_per_sentence = self.loss / self.sents
        loss_per_word = self.loss / self.words
        sent_per_second = float(self.sents) / one_time
        word_per_second = float(self.words) / one_time
        return ("Recoder <%s>, %.3f(time)/%s(updates)/%.3f(sents)/%.3f(words)/%.3f(sl-loss)/%.3f(w-loss)/%.3f(s-sec)/%.3f(w-sec)" % (self.name, one_time, self.updates, self.sents, self.words, loss_per_sentence, loss_per_word, sent_per_second, word_per_second))

    def report(self, s=""):
        utils.zlog(s+self.state(), func="info")

class MTTrainer(Trainer):
    def __init__(self, opts, model):
        super(MTTrainer, self).__init__(opts, model)

    def _validate_ll(self, dev_iter):
        # log likelihood
        one_recorder = self._get_recorder("VALID-LL")
        for insts in dev_iter.arrange_batches():
            loss = self._mm.fb(insts, False)
            one_recorder.record(insts, loss, 0)
        one_recorder.report()
        return -1 * (one_recorder.loss / one_recorder.words)

    def _validate_bleu(self, dev_iter):
        # bleu score
        mt_decode(dev_iter, [self._mm], self._mm.target_dict, self.opts, self.opts["dev_output"])
        # no restore specifies for the dev set
        s = mt_eval.evaluate(self.opts["dev_output"], self.opts["dev"][1], self.opts["eval_metric"], True)
        return s

    def _validate_them(self, dev_iter, metrics):
        validators = {"ll": self._validate_ll, "bleu": self._validate_bleu}
        r = []
        for m in metrics:
            r.append(validators[m](dev_iter))
        return ValidResult(r)

    def _get_recorder(self, name):
        return OnceRecorder(name)

    def _fb_once(self, insts):
        return self._mm.fb(insts, True)

def mt_decode(test_iter, mms, target_dict, opts, outf):
    one_recorder = OnceRecorder("DECODE")
    num_sents = len(test_iter)
    cur_sents = 0.
    # decoding them all
    results = []
    prev_point = 0
    for insts in test_iter.arrange_batches():
        if opts["verbose"] and (cur_sents - prev_point) >= (opts["report_freq"]*test_iter.bsize()):
            utils.zlog("Decoding process: %.2f%%" % (cur_sents / num_sents * 100))
            prev_point = cur_sents
        cur_sents += len(insts)
        if opts["beam_size"] == 1:
            rs = mt_search.search_greedy(mms, insts, target_dict, opts)
        else:
            # todo: currently only greedy
            raise NotImplementedError()
        results += [[int(x) for x in r[0].get_path("action")] for r in rs]
        one_recorder.record(insts, 0, 0)
    # restore from sorting by length
    results = test_iter.restore_order(results)
    with utils.zopen(outf, "w") as f:
        for r in results:
            best_seq = r
            strs = data.Vocab.i2w(target_dict, best_seq)
            f.write(" ".join(strs)+"\n")
    one_recorder.report()
