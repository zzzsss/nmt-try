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

    def _refresh(self, training, **argvs):
        # default: ingraph=True, update=True
        def _gd(drop):  # get dropout
            return drop if training else 0.
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
    def _prepare_x(self, xs):
        # batch-step-factors => step-factors-batch (using max-step, padding with <pad>)
        bsize, steps, factors = len(xs), max([len(i) for i in xs]), self.opts["factors"]
        x = [[[] for f in range(factors)] for s in range(steps)]
        lens = [len(i) for i in xs]
        for s in range(steps):
            for f in range(factors):
                padding = self.source_dicts[f].pad
                for b in range(bsize):
                    x[s][f].append(xs[b][s][f] if s<lens[b] else padding)
        return x, lens

    def _prepare_y(self, ys):
        bsize, steps = len(ys), max([len(i) for i in ys])
        y = [[] for s in range(steps)]
        lens = [len(i) for i in ys]
        padding = self.target_dict.pad
        for s in range(steps):
            for b in range(bsize):
                y[s].append(ys[b][s] if s<lens[b] else padding)
        return y, lens

    # helper routines #
    def _get_embeddings_step(self, tokens, embeds):
        # tokens: factors-batch or batch
        if type(tokens[0]) != list:
            tokens, embeds = [tokens], [embeds]
        outputs = []
        for toks, embs in zip(tokens, embeds):
            outputs.append(embs(toks))
        if len(outputs) == 1:
            return outputs[0]
        else:
            return dy.concatenate(outputs)

    def _get_embeddings(self, all_tokens, embeds):
        return [self._get_embeddings_step(s, embeds) for s in all_tokens]

    # main routines #
    def fb(self, xs, ys, backward):
        assert len(xs) == len(ys)
        bsize = len(xs)
        self._refresh(True, bsize=bsize)      # bsize for gdrop of rec-nodes
        # -- prepare batches
        xx, _ = self._prepare_x(xs)
        yy, y_lens = self._prepare_y(ys)
        # -- embeddings
        x_embeds = self._get_embeddings(xx, self.embeds_src)
        y_embeds = self._get_embeddings(yy, self.embed_trg)
        # -- encoder
        x_ctx = self.enc(x_embeds)
        # -- decoder
        self.dec.start_one(x_ctx)
        self.dec.feed_one(y_embeds)
        atts, hiddens = self.dec.get_results()
        # -- ouptuts
        losses = []
        for ss, at, hi, ye, yt in zip(range(len(yy)), atts, hiddens, y_embeds, yy):
            output_concat = dy.concatenate([at, hi, ye])
            output_score = self.outputs[1](self.outputs[0](output_concat))
            one_loss = dy.pickneglogsoftmax_batch(output_score, yt)
            mask_expr = dy.inputVector([(1. if ss<y_lens[i] else 0.) for i in range(bsize)])
            mask_expr = dy.reshape(mask_expr, (1, ), bsize)
            one_loss = one_loss * mask_expr
            losses.append(one_loss)
        # -- final
        loss = dy.esum(losses)
        loss = dy.sum_batches(loss) / bsize
        loss_val = loss.value()
        if backward:
            loss.backward()
        return loss_val*bsize

    # save and load #
    def load(self, fname):
        self.model.populate(fname)
        utils.printing("Read Model from %s." % fname, func="io")

    def save(self, fname):
        self.model.save(fname)
        utils.printing("Save Model to %s." % fname, func="io")
