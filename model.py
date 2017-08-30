# the specifications of the model
from layers import *
import utils

class NMTModel(object):
    def __init__(self, opts, source_dicts, target_dict):
        utils.printing("Start to create Model.")
        self.opts = opts
        self.source_dicts = source_dicts
        self.target_dict = target_dict
        # init models
        self.model = dy.Model()
        # embeddings
        self.embeds_src = [Embedding(self.model, len(t), i, dropout_wordceil=t.get_num_words()) for t, i in zip(source_dicts, opts["dim_per_factor"])]
        self.embed_trg = Embedding(self.model, len(target_dict), opts["dim_word"], dropout_wordceil=target_dict.get_num_words())
        # enc-dec
        self.enc = Encoder(self.model, sum(opts["dim_per_factor"]), opts["hidden_rec"], opts["enc_depth"])
        self.dec = {"att": AttDecoder, "nematus": NematusDecoder}[opts["dec_type"]](self.model, opts["dim_word"], opts["hidden_rec"], opts["dec_depth"], 2*opts["hidden_rec"], opts["hidden_att"], opts["att_type"])
        # deep output
        self.outputs = [Linear(self.model, 3*opts["hidden_rec"]+opts["dim_word"], opts["hidden_out"]),
                       Linear(self.model, opts["hidden_out"], len(target_dict), act="linear")]  # no softmax here
        utils.printing("End of creating Model.")

    def refresh(self, training, renew_cg=True, **argvs):
        # default: ingraph=True, update=True
        def _gd(drop):  # get dropout
            return drop if training else 0.
        if renew_cg:
            dy.renew_cg()   # new graph
        opts = self.opts
        # embeddings
        for e in self.embeds_src:
            e.refresh(hdrop=_gd(opts["drop_embedding"]), gdrop=_gd(opts["gdrop_embedding"]))
        self.embed_trg.refresh(hdrop=_gd(opts["drop_embedding"]), gdrop=_gd(opts["gdrop_embedding"]))
        # enc-dec
        self.enc.refresh(idrop=_gd(opts["drop_enc"]), gdrop=_gd(opts["gdrop_enc"]), **argvs)
        self.dec.refresh(idrop=_gd(opts["drop_dec"]), gdrop=_gd(opts["gdrop_dec"]), hdrop=_gd(opts["drop_hidden"]), **argvs)
        # outputs
        self.outputs[0].refresh(hdrop=_gd(opts["drop_hidden"]))
        self.outputs[1].refresh()

    # data helpers #
    def prepare_x(self, xs, fix_len):
        # batch-step-factors => step-factors-batch (using max-step, padding with <pad>)
        bsize, steps, factors = len(xs), max([len(i) for i in xs]), self.opts["factors"]
        if fix_len > 0:
            steps = fix_len
        x = [[[] for f in range(factors)] for s in range(steps)]
        lens = [len(i) for i in xs]
        for s in range(steps):
            for f in range(factors):
                padding = self.source_dicts[f].pad
                for b in range(bsize):
                    x[s][f].append(xs[b][s][f] if s<lens[b] else padding)
        # check last step
        for f in range(factors):
            eos, padding = self.source_dicts[f].eos, self.source_dicts[f].pad
            for b in range(bsize):
                if x[-1][f][b] not in [eos, padding]:
                    x[-1][f][b] = eos
        return x, lens

    def prepare_y(self, ys, fix_len):
        bsize, steps = len(ys), max([len(i) for i in ys])
        if fix_len > 0:
            steps = fix_len
        y = [[] for s in range(steps)]
        lens = [len(i) for i in ys]
        padding = self.target_dict.pad
        for s in range(steps):
            for b in range(bsize):
                y[s].append(ys[b][s] if s<lens[b] else padding)
        # check last step
        eos, padding = self.target_dict.eos, self.target_dict.pad
        for b in range(bsize):
            if y[-1][b] not in [eos, padding]:
                y[-1][b] = eos
        return y, lens

    # helper routines #
    def get_embeddings_step(self, tokens, embeds):
        # factors-batch([[w1, w2, ...], [p1, p2, ...], ...]) or batch([w1, w2, ...]) or int(w1)
        if type(tokens) == int:    # int => [int]
            tokens = [tokens]
        if type(tokens[0]) != list:
            tokens, embeds = [tokens], [embeds]
        outputs = []
        for toks, embs in zip(tokens, embeds):
            outputs.append(embs(toks))
        if len(outputs) == 1:
            return outputs[0]
        else:
            return dy.concatenate(outputs)

    def get_embeddings(self, all_tokens, embeds):
        return [self.get_embeddings_step(s, embeds) for s in all_tokens]

    def get_start_yembs(self, bsize):
        return self.get_embeddings_step([self.target_dict.start for _ in range(bsize)], self.embed_trg)

    def get_score(self, at, hi, ye):
        output_concat = dy.concatenate([at, hi, ye])
        output_score = self.outputs[1](self.outputs[0](output_concat))
        return output_score

    # main routines #
    def fb(self, xs, ys, training):
        assert len(xs) == len(ys)
        bsize = len(xs)
        self.refresh(training, bsize=bsize)      # bsize for gdrop of rec-nodes
        # -- prepare batches
        xx, _ = self.prepare_x(xs, self.opts["fix_len_src"])
        yy, y_lens = self.prepare_y(ys, self.opts["fix_len_trg"])
        # -- embeddings
        x_embeds = self.get_embeddings(xx, self.embeds_src)
        y_embeds = self.get_embeddings(yy, self.embed_trg)
        # --- shift y_embeddings (ignore the last one which must be eos)
        y_yes = [self.get_start_yembs(bsize)] + y_embeds[:-1]
        # -- encoder
        x_ctx = self.enc(x_embeds)
        # -- decoder (steps == len)
        ss = self.dec.start_one(x_ctx)
        ss = self.dec.feed_one(ss, y_embeds[:-1])
        hiddens, atts, _ = ss.get_results()
        # -- ouptuts
        losses = []
        for ss, at, hi, ye, yt in zip(range(len(yy)), atts, hiddens, y_yes, yy):
            output_score = self.get_score(at, hi, ye)
            one_loss = dy.pickneglogsoftmax_batch(output_score, yt)
            mask_expr = dy.inputVector([(1. if ss<y_lens[i] else 0.) for i in range(bsize)])
            mask_expr = dy.reshape(mask_expr, (1, ), bsize)
            one_loss = one_loss * mask_expr
            losses.append(one_loss)
        # -- final
        loss = dy.esum(losses)
        loss = dy.sum_batches(loss) / bsize
        loss_val = loss.value()
        if training:
            loss.backward()
        return loss_val*bsize

    def fb2(self, xs, ys, training):
        # for debugging, should be the same as fb(training=False)
        assert len(xs) == len(ys)
        self.refresh(False)
        loss = 0.
        for x, y in zip(xs, ys):
            ss = None
            for i in range(len(y)):
                if i==0:
                    ye = self.get_start_yembs(1)
                    ss = self.prepare_enc([x], 1)
                else:
                    ye = self.get_embeddings_step(y[i-1], self.embed_trg)
                    ss = self.dec.feed_one(ss, ye)
                hi, at, atw = ss.get_results_one()
                sc = self.get_score(at, hi, ye)
                prob = dy.softmax(sc)
                pvalue = prob.value()
                loss -= np.log(pvalue[y[i]])
        return loss

    # for predciting #
    def prepare_enc(self, xs, expand):
        # -- encode & start-decode: return list of DecoderState
        mm = self
        xx = mm.prepare_x(xs, -1)[0]
        x_embed = mm.get_embeddings(xx, mm.embeds_src)
        x_ctx = mm.enc(x_embed)
        ss = mm.dec.start_one(x_ctx, expand)
        return ss

    # save and load #
    def load(self, fname):
        self.model.populate(fname)
        utils.printing("Read Model from %s." % fname, func="io")

    def save(self, fname):
        self.model.save(fname)
        utils.printing("Save Model to %s." % fname, func="io")
