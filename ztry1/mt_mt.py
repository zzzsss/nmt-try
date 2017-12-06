# focusing on speed

from zl.model import Model
from zl import utils, data
from .mt_length import MTLengther, LinearGaussain
import numpy as np
from . import mt_layers as layers

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
        # other model properties
        if opts["train_r2l"]:
            self.set_prop("r2l", True)
        if opts["no_model_softmax"]:
            self.set_prop("no_model_softmax", True)
        self.fber_ = self.fb_standard_
        #
        self.penalize_eos = opts["penalize_eos"]
        self.penalize_list = self.target_dict.get_ending_tokens()

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

    def recombine(self, clist, idxlist):
        new_c = {}
        for names in (self.names_bv, self.names_bi):
            for n in names:
                new_c[n] = layers.BK.recombine_cache([_c[n] for _c in clist], idxlist)
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
    def start(self, xs, repeat_time=1, softmax=True):
        # encode
        bsize = len(xs)
        xx, xm = prepare_data(xs, self.source_dict)
        x_encodes = self.encode(xx, xm)
        x_encodes = [layers.BK.batch_repeat(one, repeat_time) for one in x_encodes]
        # init decode
        cache = self.decode_start(x_encodes)
        start_embeds = self.get_start_yembs(bsize)
        output_score, output_hidden = self.get_scores(cache["ctx"], cache["hid"], start_embeds)
        if softmax and not self.get_prop("no_model_softmax"):
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
        if softmax and not self.get_prop("no_model_softmax"):
            results = layers.BK.softmax(output_score)
        else:
            results = output_score
        # return
        cache["out_s"] = output_score
        cache["results"] = results
        return cache

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

    # for the results
    def explain_result(self, x, one_idx=None):
        if self.get_prop("no_model_softmax"):
            ss = x
        else:
            ss = np.log(x)
        # todo(warn): penalizing here for convenience
        if self.penalize_eos > 0.:
            if one_idx is not None:
                if one_idx in self.penalize_list:
                    ss -= self.penalize_eos
            else:
                for idx in self.penalize_list:
                    ss[idx] -= self.penalize_eos
        return ss

    # training
    def fb(self, insts, training, ret_value="loss", new_graph=True):
        if new_graph:
            Model.new_graph()
            self.refresh(training)
        r = self.fber_(insts, training, ret_value)
        return r

    def fb_standard_(self, insts, training, ret_value="loss"):
        xs = [i[0] for i in insts]
        if self.get_prop("r2l"):
            # right to left modeling, be careful about eos
            ys = [list(reversed(i[1][:-1]))+[i[1][-1]] for i in insts]
        else:
            ys = [i[1] for i in insts]
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
        if ret_value == "loss":
            lossy_val = layers.BK.get_value_sca(loss_y)
            loss_len_val = -1.
            if self.is_fitting_length:
                loss_len_val = layers.BK.get_value_sca(loss_len)
            return {"y": lossy_val*bsize, "len": loss_len_val}
        elif ret_value == "losses":
            # return token-wise loss
            origin_values = [layers.BK.get_value_vec(i) for i in losses]
            reshaped_values = [[origin_values[j][i] for j in range(yl)] for i, yl in enumerate(ylens)]
            if self.get_prop("r2l"):
                reshaped_values = [list(reversed(one[:-1]))+[one[-1]] for one in reshaped_values]
            return reshaped_values
        else:
            return {}

# ----- losses -----

def _mle_loss_step(scores_exprs, ystep, mask_expr):
    one_loss = layers.BK.pickneglogsoftmax_batch(scores_exprs, ystep)
    if mask_expr is not None:
        one_loss = one_loss * mask_expr
    return one_loss

# ----- losses -----
