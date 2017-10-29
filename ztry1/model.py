from zl.model import Model
from zl import layers, utils

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
        # What is in the cache: S,V/ cov,ctx,att,/ out_hid,/ out_s,results => these are handled/rearranged by the Scorer
        self.names_bv = {"cov", "ctx", "hid", "out_hid"}
        self.names_bi = {"S", "V"}
        self.names_ig = {"att", "out_s", "results"}
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

    # data helpers #
    def prepare_data(self, ys, fix_len=0):
        # input: list of list of index (bsize, step),
        # output: padded input (step, bsize), masks (1 for real words, 0 for paddings)
        bsize, steps = len(ys), max([len(i) for i in ys])
        if fix_len > 0:
            steps = fix_len
        y = [[] for _ in range(steps)]
        lens = [len(i) for i in ys]
        eos, padding = self.target_dict.eos, self.target_dict.pad
        for s in range(steps):
            for b in range(bsize):
                y[s].append(ys[b][s] if s<lens[b] else padding)
        # check last step
        for b in range(bsize):
            if y[-1][b] not in [eos, padding]:
                y[-1][b] = eos
        return y, [[(1. if len(one)>s else 0.) for one in ys] for s in range(steps)]

    # helper routines #
    def get_embeddings_step(self, tokens, embed):
        # tokens: list of int or one int, embed: Embedding => one expression (batched)
        return embed(tokens)

    def get_start_yembs(self, bsize):
        return self.get_embeddings_step([self.target_dict.bos for _ in range(bsize)], self.embed_trg)

    def get_scores(self, at, hi, ye):
        output_concat = layers.BK.concatenate([at, hi, ye])
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
    def start(self, xs, repeat=1):
        # encode
        bsize = len(xs)
        xx, xm = self.prepare_data(xs)
        x_encodes = self.encode(xx, xm)
        x_encodes = [layers.BK.batch_repeat(one, repeat) for one in x_encodes]
        # init decode
        cache = self.decode_start(x_encodes)
        start_embeds = self.get_start_yembs(bsize)
        output_score, output_hidden = self.get_scores(cache["ctx"], cache["hid"], start_embeds)
        probs = layers.BK.softmax(output_score)
        return utils.Helper.combine_dicts(cache, {"out_hid": output_hidden, "out_s": output_score, "results": probs})

    def step(self, prev_val, inputs):
        x_encodes, hiddens = None, prev_val["hid"]
        next_embeds = self.get_embeddings_step(inputs, self.embed_trg)
        cache = self.decode_step(x_encodes, inputs, next_embeds, prev_val)
        output_score, output_hidden = self.get_scores(cache["ctx"], cache["hid"], next_embeds)
        probs = layers.BK.softmax(output_score)
        return utils.Helper.combine_dicts(cache, {"out_hid": output_hidden, "out_s": output_score, "results": probs})
