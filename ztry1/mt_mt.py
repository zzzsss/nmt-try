# focusing on speed

from zl.model import Model
from zl import utils, data
from .mt_length import MTLengther, LinearGaussain
import numpy as np
from . import mt_layers as layers
from zl.search import State, SearchGraph, Action
from .mt_search import Pruner as SPruner

from zl.backends.common import ResultTopk
IDX_DIM = ResultTopk.IDX_DIM
VAL_DIM = ResultTopk.VAL_DIM

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
        self.dec_ngram_n = opts["dec_ngram_n"]
        if opts["dec_type"] == "ngram":
            self.dec = layers.NgramDecoder(self.model, opts["dim_word"], opts["hidden_dec"], opts["dec_depth"], 2*opts["hidden_enc"],
                    opts["hidden_att"], opts["att_type"], opts["rnn_type"], opts["summ_type"], self.dec_ngram_n)
        else:
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
        self.fber_ = {"std2":self.fb_standard2_, "beam":self.fb_beam_, "branch":self.fb_branch_}[opts["train_mode"]]
        self.losser_ = {"mle":self._mle_loss_step, "mlev":self._mlev_loss_step, "hinge_max":self._hinge_max_loss_step,
                        "hinge_avg":self._hinge_avg_loss_step, "hinge_sum":self._hinge_sum_loss_step}[opts["train_local_loss"]]
        self.margin_ = opts["train_margin"]
        utils.zlog("For the training process %s, using %s; loss is %s, using %s; margin is %s"
                   % (opts["train_mode"], self.fber_, opts["train_local_loss"], self.losser_, self.margin_))
        #
        self.penalize_eos = opts["penalize_eos"]
        self.penalize_list = self.target_dict.get_ending_tokens()

    def repeat(self, c, bs, times, names):
        new_c = {}
        orders = [i//times for i in range(bs*times)]
        for n in names:
            new_c[n] = layers.BK.rearrange_cache(c[n], orders)
        return new_c

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
        bos = self.target_dict.bos
        return self.get_embeddings_step([bos for _ in range(bsize)], self.embed_trg)

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

    def decode_step(self, x_encodes, inputs, caches, prev_embeds):
        # feed one step
        return self.dec.feed_one(x_encodes, inputs, caches, prev_embeds)

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

    def step(self, prev_val, inputs, cur_states=None, softmax=True):
        x_encodes, hiddens = None, prev_val["hid"]
        next_embeds = self.get_embeddings_step(inputs, self.embed_trg)
        # prepare prev_embeds
        if self.opts["dec_type"] == "ngram":
            bos = self.target_dict.bos
            prev_tokens = [s.sig_ngram_tlist(self.dec_ngram_n, bos) for s in cur_states]
            prev_embeds = [self.get_embeddings_step([p[step] for p in prev_tokens], self.embed_trg) for step in range(self.dec_ngram_n)]
        else:
            prev_embeds = None
        cache = self.decode_step(x_encodes, next_embeds, prev_val, prev_embeds)
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

    # for the results: input is pairs(K) of (idx, val)
    def explain_result_topkp(self, pairs):
        to_log = not self.get_prop("no_model_softmax")
        to_penalize = self.penalize_eos > 0.
        ret_pairs = []
        for p in pairs:
            if to_log:
                p[VAL_DIM] = np.log(p[VAL_DIM])
            if to_penalize:
                if p[IDX_DIM] in self.penalize_list:
                    p[VAL_DIM] -= self.penalize_eos
            if p[IDX_DIM] != self.target_dict.err:
                ret_pairs.append(p)
        # re-sorting
        ret_pairs.sort(key=lambda p: p[VAL_DIM], reverse=True)
        return ret_pairs

    # =============================
    # training
    def fb(self, insts, training, ret_value="loss", new_graph=True):
        if new_graph:
            Model.new_graph()
            self.refresh(training)
        r = self.fber_(insts, training, ret_value)
        return r

    # helpers
    def prepare_xy_(self, insts):
        xs = [i[0] for i in insts]
        if self.get_prop("r2l"):
            # right to left modeling, be careful about eos
            ys = [list(reversed(i[1][:-1]))+[i[1][-1]] for i in insts]
        else:
            ys = [i[1] for i in insts]
        return xs, ys

    # specific forward/backward runs
    def fb_standard2_(self, insts, training, ret_value="loss"):
        # please don't ask me where is standard1 ...
        # similar to standard1, but record states and no training for lengths
        xs, ys = self.prepare_xy_(insts)
        bsize = len(xs)
        opens = [State(sg=SearchGraph(target_dict=self.target_dict)) for _ in range(bsize)]     # with only one init state
        # xlens = [len(_x) for _x in xs]
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
            else:
                cc = self.step(caches[-1], yprev, opens, softmax=False)
            caches.append(cc)
            yprev = ystep
            # build loss
            scores_exprs = cc["out_s"]
            loss = self.losser_(scores_exprs, ystep, mask_expr)
            if self.scaler is not None:
                len_scales = [self.scaler(len(_y)) for _y in ys]
                np.asarray(len_scales).reshape((1, -1))
                len_scalue_e = layers.BK.inputTensor(len_scales, True)
                loss = loss * len_scalue_e
            losses.append(loss)
            # prepare next steps: only following gold
            new_opens = [State(prev=ss, action=Action(yy, 0.)) for ss, yy in zip(opens, ystep)]
            opens = new_opens
        # -- final
        loss0 = layers.BK.esum(losses)
        loss = layers.BK.sum_batches(loss0) / bsize
        if training:
            layers.BK.forward(loss)
            layers.BK.backward(loss)
        # return value?
        if ret_value == "loss":
            lossy_val = layers.BK.get_value_sca(loss)
            return {"y": lossy_val*bsize}
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

    def _mlev_loss_step(self, scores_exprs, ystep, mask_expr):
        # (wasted) getting values to test speed
        # gold_exprs = layers.BK.pick_batch(scores_exprs, ystep)
        # gold_vals = layers.BK.get_value_vec(gold_exprs)
        # pp = layers.BK.topk(scores_exprs, 1)
        pp = layers.BK.topk(scores_exprs, 8)
        # max_exprs = layers.BK.max_dim(scores_exprs)
        # scores_exprs.forward()
        # max_tenidx0 = scores_exprs.tensor_value()
        # zz = max_tenidx0.argmax()
        # max_tenidx = scores_exprs.tensor_value().argmax()
        return self._mle_loss_step(scores_exprs, ystep, mask_expr)

    def _mle_loss_step(self, scores_exprs, ystep, mask_expr):
        if self.margin_ > 0.:
            scores_exprs = layers.BK.add_margin(scores_exprs, ystep, self.margin_)
        one_loss = layers.BK.pickneglogsoftmax_batch(scores_exprs, ystep)
        if mask_expr is not None:
            one_loss = one_loss * mask_expr
        return one_loss

    def _hinge_max_loss_step(self, scores_exprs, ystep, mask_expr):
        if self.margin_ > 0.:
            scores_exprs_final = layers.BK.add_margin(scores_exprs, ystep, self.margin_)
        else:
            scores_exprs_final = scores_exprs
        # max_exprs = layers.BK.max_dim(scores_exprs)
        max_idxs = layers.BK.topk(scores_exprs_final, 1, prepare=False)[IDX_DIM]
        max_exprs = layers.BK.pick_batch(scores_exprs_final, max_idxs)
        gold_exprs = layers.BK.pick_batch(scores_exprs_final, ystep)
        # get loss
        one_loss = max_exprs - gold_exprs
        if mask_expr is not None:
            one_loss = one_loss * mask_expr
        return one_loss

    def _hinge_avg_loss_step(self, scores_exprs, ystep, mask_expr):
        one_loss_all = layers.BK.hinge_batch(scores_exprs, ystep, self.margin_)
        ## todo: approximate counting code here (-3 for squeezing negatives, +1 for smoothing)
        # -- still out of memory on 12g gpu, maybe need smaller bs
        gold_exprs = layers.BK.pick_batch(scores_exprs, ystep)
        gold_exprs -= (self.margin_ - 3)    # to get to zero for sigmoid
        scores_exprs -= gold_exprs
        count_exprs = layers.BK.sum_elems(layers.BK.logistic(scores_exprs))
        count_exprs = layers.BK.nobackprop(count_exprs)
        one_loss = layers.BK.cdiv(one_loss_all, count_exprs+1)
        if mask_expr is not None:
            one_loss = one_loss * mask_expr
        return one_loss

    def _hinge_sum_loss_step(self, scores_exprs, ystep, mask_expr):
        one_loss = layers.BK.hinge_batch(scores_exprs, ystep, self.margin_)
        if mask_expr is not None:
            one_loss = one_loss * mask_expr
        return one_loss

    # ----- losses -----

    # todo(warn): advanced training process, similar to the mt_search part, but rewrite to avoid messing up, which
    # -> is really not a good idea, and this should be combined with mt_search, however ...
    # -> do not re-use the decoding part of opts, rename those to t2_*

    def fb_beam_(self, insts, training, ret_value):
        # always keeping the same size for more efficient batching
        utils.zcheck(training, "Only for training mode.")
        xs, ys = self.prepare_xy_(insts)
        # --- no need to refresh
        # -- options and vars
        # basic
        bsize = len(xs)
        esize_all = self.opts["t2_beam_size"]
        ylens = [len(_y) for _y in ys]
        ylens_max = [int(np.ceil(_yl*self.opts["t2_search_ratio"])) for _yl in ylens]   # todo: currently, just cut off according to ref length
        # pruners (no diversity penalty here for training)
        # -> local
        t2_local_expand = min(self.opts["t2_local_expand"], esize_all)
        t2_local_diff = self.opts["t2_local_diff"]
        # -> global beam/gold merging for ngram
        t2_global_expand = self.opts["t2_global_expand"]
        t2_global_diff = self.opts["t2_global_diff"]
        t2_bngram_n = self.opts["t2_bngram_n"]
        t2_bngram_range = self.opts["t2_bngram_range"]
        t2_gngram_n = self.opts["t2_gngram_n"]
        t2_gngram_range = self.opts["t2_gngram_range"]
        #
        # specific running options
        CACHE_NAME = "CC"
        PADDING_STATE = None
        PADDING_ACTION = 0
        EOS_ID = self.target_dict.eos
        model_softmax = not self.get_prop("no_model_softmax")
        t2_gold_run = self.opts["t2_gold_run"]

        # => START
        State.reset_id()
        f_new_sg = lambda _i: SearchGraph(target_dict=self.target_dict, src_info=insts[_i])
        # 1. first running the gold seqs if needed
        gold_cur = [State(sg=f_new_sg(_i)) for _i in range(bsize)]
        gold_states = [gold_cur]
        running_yprev = None
        running_cprev = None
        for step_gold in range(max(ylens)):
            # running the caches
            ystep, mask_expr = prepare_y_step(ys, step_gold)    # set 0 for padding if ended
            # select the next steps anyway
            gold_next = []
            for _i in range(bsize):
                x = gold_cur[_i]
                if x is PADDING_STATE or x.action_code == EOS_ID:
                    gold_next.append(PADDING_STATE)
                else:
                    gold_next.append(State(prev=x, action=Action(ystep[_i], 0.)))
            if t2_gold_run:
                # softmax for the "correct" score if needed
                if step_gold == 0:
                    cc = self.start(xs, softmax=model_softmax)
                else:
                    cc = self.step(running_cprev, running_yprev, gold_cur, softmax=model_softmax)
                running_yprev = ystep
                running_cprev = cc
                # attach caches
                sc_expr = layers.BK.pick_batch(cc["results"], ystep)
                sc_val = layers.BK.get_value_vec(sc_expr)
                # todo(warn): here not calling explain for simplicity
                if model_softmax:
                    sc_val = np.log(sc_val)
                for _i in range(bsize):
                    if gold_cur[_i] is not PADDING_STATE:
                        gold_cur[_i].set(CACHE_NAME, (cc, _i))
                        gold_cur[_i].action_score(sc_val[_i])
            gold_cur = gold_next
            gold_states.append(gold_next)
        # 2. then start the beam search (with the knowledge of gold-seq)
        beam_states = []
        beam_cur = [[PADDING_STATE for _j in range(esize_all)] for _i in range(bsize)]
        beam_states.append(beam_cur)
        beam_ends = []
        beam_remains = [esize_all for _i in range(bsize)]
        for _i in range(bsize):     # new search graph
            beam_cur[_i][0] = State(sg=f_new_sg(_i))
        # ready go, break at the end
        running_yprev = None
        running_cprev = None
        step_beam = 0
        while True:
            # running the caches
            if step_beam == 0:
                if t2_gold_run:
                    # todo(warn): expand previously calculated values and also results
                    cc = self.repeat(gold_states[0][0].get(CACHE_NAME)[0], bsize, esize_all, {"hid", "S", "V", "out_s", "results"})
                else:
                    cc = self.start(xs, repeat_time=esize_all, softmax=model_softmax)
            else:
                cc = self.step(running_cprev, running_yprev, gold_cur, softmax=model_softmax)
            # attach caches
            for _i in range(bsize):
                for _j in range(esize_all):
                    one = beam_cur[_i][_j]
                    if one is not PADDING_STATE:
                        one.set(CACHE_NAME, (cc, _i*esize_all+_j))
            # compare for the next steps --- almost same as beam_search, but all-batched and simplified
            results_topk = layers.BK.topk(cc["results"], t2_local_expand)
            for i in range(bsize):
                # collect local candidates
                global_cands = []
                for j in range(esize_all):
                    prev = beam_cur[i][j]
                    # skip ended states
                    if prev is PADDING_STATE or prev.action_code == EOS_ID:
                        continue
                    inbatch_idx = i*esize_all+j
                    rr0 = results_topk[inbatch_idx]
                    rr = self.explain_result_topkp(rr0)
                    local_cands = []
                    for onep in rr:
                        local_cands.append(State(prev=prev, action=Action(onep[IDX_DIM], onep[VAL_DIM])))
                    survived_local_cands = SPruner.local_prune(local_cands, t2_local_expand, t2_local_diff, 0.)
                    global_cands += survived_local_cands
                # sort them all
                global_cands.sort(key=(lambda x: x.score_partial), reverse=True)
                # global pruning (if no bngram, then simply get the first remains[i] ones)
                survived_global_cands = SPruner.global_prune_ngram_greedy(cand_states=global_cands, rest_beam_size=beam_remains[i], sig_beam_size=t2_global_expand, thresh=t2_global_diff, penalty=0., ngram_n=t2_bngram_n, ngram_range=t2_bngram_range)
                # ======================
                # gold checking

            # todo(assign them)
            running_yprev = ystep
            running_cprev = cc
            # how to break?
            step_beam += 1
        # -- recorders
        return

    def fb_branch_(self, insts, training, ret_value):
        # this branches from gold which is different from the search_branch_ which branches from greedy-best
        raise NotImplementedError("To be implemented")
        pass
