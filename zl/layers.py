# the general components of a neural model
# -- similar to Keras, but with dynamic toolkit as backends

from . import utils
from collections import Iterable
import numpy
from .backends import BK_DY as BK

# ================= Basic Blocks ================= #
# basic unit (stateful about dropouts)
class Layer(object):
    def __init__(self, model):
        # basic ones: mainly the parameters
        self.model = model
        self.params = {}
        self.iparams = {}
        self.update = None
        # aux info like dropouts/masks (could be refreshed)
        self.hdrop = 0.     # hidden output drops
        self.idrop = 0.     # input drops
        self.gdrop = 0.     # special recurrent drops
        self.gmasks = None  # special masks for gdrop (pre-defined drops)
        self.bsize = None   # bsize for one f/b

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("No calling __call__ from Layer.")

    def __repr__(self):
        return "Basic Layer"

    def __str__(self):
        return self.__repr__()

    def refresh(self, argv):
        self._refresh(argv)

    # !! this one should be FINAL, not overridden
    def _refresh(self, argv):
        # update means whether the parameters should be updated
        update = bool(argv["update"]) if "update" in argv else True
        ingraph = bool(argv["ingraph"]) if "ingraph" in argv else True
        if ingraph:
            for k in self.params:
                # todo(warn): special convention
                if not k.startswith("_"):
                    self.iparams[k] = BK.param2expr(self.params[k], update)
            self.update = update
        # dropouts
        self.hdrop = float(argv["hdrop"]) if "hdrop" in argv else 0.
        self.idrop = float(argv["idrop"]) if "idrop" in argv else 0.
        self.gdrop = float(argv["gdrop"]) if "gdrop" in argv else 0.
        self.gmasks = None
        self.bsize = int(argv["bsize"]) if "bsize" in argv else None

    def _add_params(self, shape, lookup=False, init="default"):
        return BK.get_params(self.model, shape, lookup, init)

# linear layer with selectable activation functions
# [inputs] or input -> output
class Affine(Layer):
    _ACTS = ["linear", "tanh", "softmax"]
    _ACT_DEFAULT = "linear"

    def __init__(self, model, n_ins, n_out, act="tanh", bias=True):
        super(Affine, self).__init__(model)
        # list of n_ins and n_outs have different meanings: horizontal and vertical
        if not isinstance(n_ins, Iterable):
            n_ins = [n_ins]
        # dimensions
        self.n_ins = n_ins
        self.n_out = n_out
        # activations
        self.act = act
        self._act_ffs = {"tanh":BK.tanh, "softmax":BK.softmax, "linear":lambda x:x}[self.act]
        # params
        self.bias = bias
        for i, din in enumerate(n_ins):
            self.params["W"+str(i)] = self._add_params((n_out, din))
        if bias:
            self.params["B"] = self._add_params((n_out,))

    def __repr__(self):
        return "# Affine (%s -> %s [%s])" % (self.n_ins, self.n_out, self.act)

    def __call__(self, input_exp):
        if not isinstance(input_exp, Iterable):
            input_exp = [input_exp]
        if self.bias:
            input_lists = [self.iparams["B"]]
        else:
            input_lists = [BK.zeros(self.n_out)]
        for i, one_inp in enumerate(input_exp):
            input_lists += [self.iparams["W"+str(i)], one_inp]
        h0 = BK.affine(input_lists)
        h1 = self._act_ffs(h0)
        if self.hdrop > 0.:
            h1 = BK.dropout(h1, self.hdrop)
        return h1

# nearly the same as affine but enforcing no-dropout (usually as output layer)
class AffineNodrop(Affine):
    def __init__(self, model, n_ins, n_out, act="linear", bias=True):
        super(AffineNodrop, self).__init__(model, n_ins, n_out, act, bias)

    def __repr__(self):
        return "# AffineNodrop (%s -> %s [%s])" % (self.n_ins, self.n_out, self.act)

    def __call__(self, input_exp):
        self.hdrop = 0.
        super(AffineNodrop, self).__call__(input_exp)

# embedding layer
# [inputs] or input -> (batched) output
class Embedding(Layer):
    def __init__(self, model, n_words, n_dim, dropout_wordceil=None, npvec=None):
        super(Embedding, self).__init__(model)
        if npvec is not None:
            utils.zforce(utils.zcheck, len(npvec.shape) == 2 and npvec.shape[0] == n_words and npvec.shape[1] == n_dim, "Wrong dimension for init embeddings.")
            self.params["_E"] = self._add_params((n_words, n_dim), lookup=True, init=npvec)
        else:
            self.params["_E"] = self._add_params((n_words, n_dim), lookup=True, init="random")
        self.n_dim = n_dim
        self.n_words = n_words
        self.dropout_wordceil = dropout_wordceil if dropout_wordceil is not None else n_words

    def refresh(self, argv):
        self._refresh(argv)
        # zero out (todo: for tr?)
        self.params["E"].init_row(0, [0. for _ in range(self.n_dim)])
        # special treatment (todo: for tr?)
        self.iparams["E"] = self.params["_E"]

    def __repr__(self):
        return "# Embedding (dim=%s, num=%s)" % (self.n_dim, self.n_words)

    def __call__(self, input_exp):
        # input should be a list of ints or int
        if type(input_exp) != list:
            input_exp = [input_exp]
        if self.idrop > 0:
            input_exp = [(0 if (v<=0 and i>=self.dropout_wordceil) else i) for i,v in zip(input_exp, utils.Random.binomial(1, 1-self.idrop, len(input_exp), "drop"))]
        x = BK.lookup_batch(self.iparams["E"], input_exp, self.update)   # (todo: for tr?)
        if self.hdrop > 0:
            x = BK.dropout(x, self.hdrop)
        return x

# rnn nodes
# [inputs] or input + {"H":hidden, "C":(optional)cell} -> {"H":hidden, "C":(optional)cell}
class RnnNode(Layer):
    def __init__(self, model, n_inputs, n_hidden):
        super(RnnNode, self).__init__(model)
        if not isinstance(n_inputs, Iterable):
            n_inputs = [n_inputs]
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden

    def refresh(self, argv):
        self._refresh(argv)
        if self.gdrop > 0:   # same masks for all instances in the batch
            # todo(warn): 1. gdrop for both or rec-only? 2. diff gdrop for gates or not? 3. same for batches or not?
            # special gdrops: [inputs+hidden for _ in num_gmasks]
            _tmp_ff = BK.random_bernoulli
            self.gmasks = [[_tmp_ff(self.gdrop, nn, 1) for nn in self.n_inputs] + [_tmp_ff(self.gdrop, self.n_hidden, 1)] for _ in range(self.num_gmasks())]

    def num_gmasks(self):
        return 1    # default: no diff gdrop for different components

    def __call__(self, input_exp, hidden_exp, mask):
        # todo(warn) return a {}
        raise NotImplementedError()

    def __repr__(self):
        return "# RnnNode[%s] (input=%s, hidden=%s)" % (type(self), self.n_inputs, self.n_hidden)

    @staticmethod
    def _pass_mask(hidden_origins, hidden_news, mask):
        # mask: if 0 then pass through
        mask_array = numpy.asarray(mask).reshape((1, -1))
        m1 = BK.inputTensor(mask_array, True)           # 1.0 for real words
        m0 = BK.inputTensor(1.0 - mask_array, True)     # 1.0 for padding words (mask=0)
        hiddens = [ho * m1 + hn * m0 for ho, hn in zip(hidden_origins, hidden_news)]
        return hiddens

    @staticmethod
    def get_rnode(s):
        return {"gru":GruNode, "lstm":LstmNode, "dummy":DmRnnNode}[s]

class DmRnnNode(RnnNode):
    def __init__(self, model, n_input, n_hidden):
        super(DmRnnNode, self).__init__(model, n_input, n_hidden)

    def __call__(self, input_exp, hidden_exp, mask=None):
        return hidden_exp

class GruNode(RnnNode):
    def __init__(self, model, n_inputs, n_hidden):
        super(GruNode, self).__init__(model, n_inputs, n_hidden)
        # paramters
        for i, dim in enumerate(n_inputs):
            self.params["x2r_%s"%i] = self._add_params((n_hidden, dim))
        self.params["h2r"] = self._add_params((n_hidden, n_hidden), init="ortho")
        self.params["br"] = self._add_params((n_hidden,))
        for i, dim in enumerate(n_inputs):
            self.params["x2z_%s"%i] = self._add_params((n_hidden, dim))
        self.params["h2z"] = self._add_params((n_hidden, n_hidden), init="ortho")
        self.params["bz"] = self._add_params((n_hidden,))
        for i, dim in enumerate(n_inputs):
            self.params["x2h_%s"%i] = self._add_params((n_hidden, dim))
        self.params["h2h"] = self._add_params((n_hidden, n_hidden), init="ortho")
        self.params["bh"] = self._add_params((n_hidden,))

    def num_gmasks(self):
        return 2

    def __call__(self, input_exp, hidden_exp, mask=None):
        # tmp one for convenience
        def _ff_list(s, ins):
            r = []
            for i in self.n_inputs:
                r.append(self.iparams["%s_%s"%(s,i)])
                r.append(ins[i])
            return r
        # two kinds of dropouts
        if not isinstance(input_exp, Iterable):
            input_exp = [input_exp]
        if self.idrop > 0.:
            input_exp = [BK.dropout(one, self.idrop) for one in input_exp]
        input_exp_g = input_exp_t = input_exp
        hidden_exp_g = hidden_exp_t = hidden_exp["H"]
        if self.gdrop > 0.:
            input_exp_g = [BK.cmult(one, dd) for one, dd in zip(input_exp_g, self.gmasks[0][:-1])]
            hidden_exp_g = BK.cmult(hidden_exp_g, self.gmasks[0][-1])
            input_exp_t = [BK.cmult(one, dd) for one, dd in zip(input_exp_t, self.gmasks[1][:-1])]
            hidden_exp_t = BK.cmult(hidden_exp_t, self.gmasks[1][-1])
        rt = BK.affine([self.iparams["br"], self.iparams["h2r"], hidden_exp_g] + _ff_list("x2r", input_exp_g))
        rt = BK.logistic(rt)
        zt = BK.affine([self.iparams["bz"], self.iparams["h2z"], hidden_exp_g] + _ff_list("x2z", input_exp_g))
        zt = BK.logistic(zt)
        h_reset = BK.cmult(rt, hidden_exp_t)
        ht = BK.affine([self.iparams["bh"], self.iparams["h2h"], h_reset] + _ff_list("x2h", input_exp_t))
        ht = BK.tanh(ht)
        hidden = BK.cmult(zt, hidden_exp["H"]) + BK.cmult((1. - zt), ht)     # first one use original hh
        if mask is not None:
            hidden = self._pass_mask([hidden_exp["H"]], [hidden], mask)[0]
        return {"H": hidden}

class LstmNode(RnnNode):
    def __init__(self, model, n_input, n_hidden):
        super(LstmNode, self).__init__(model, n_input, n_hidden)
        # paramters
        self.params["xw"] = self._add_params((n_hidden*4, n_input))
        self.params["hw"] = self._add_params((n_hidden*4, n_hidden), init="ortho")
        self.params["b"] = self._add_params((n_hidden*4,))

    def __call__(self, input_exp, hidden_exp, mask=None):
        # two kinds of dropouts
        if not isinstance(input_exp, Iterable):
            input_exp = [input_exp]
        if self.idrop > 0.:
            input_exp = [BK.dropout(one, self.idrop) for one in input_exp]
        if self.gdrop > 0.:     # todo(warn): only use 2 masks and only one input
            hidden, cc = BK.vanilla_lstm(input_exp, hidden_exp["H"], hidden_exp["C"], self.iparams["xw"], self.iparams["hw"], self.iparams["b"], self.gmasks[0][0], self.gmasks[0][-1])
        else:
            hidden, cc = BK.vanilla_lstm(input_exp, hidden_exp["H"], hidden_exp["C"], self.iparams["xw"], self.iparams["hw"], self.iparams["b"], None, None)
        # mask: if 0 then pass through
        if mask is not None:
            hidden, cc = self._pass_mask([hidden_exp["H"], hidden_exp["C"]], [hidden, cc], mask)
        return {"H": hidden, "C": cc}

# stateless attender
# [srcs] + target + caches -> {ctx, att}
class Attention(Layer):
    def __init__(self, model, n_src, n_trg, n_hidden, n_cov):
        super(Attention, self).__init__(model)
        self.n_src, self.n_trg, self.n_hidden, self.n_cov = n_src, n_trg, n_hidden, n_cov
        self.cov_gru = None

    def __repr__(self):
        return "# Attention[%s] (src=%s, trg=%s, hidden=%s, cov=%s)" % (type(self), self.n_src, self.n_trg, self.n_hidden, self.n_cov)

    def __call__(self, s, n, caches):
        # (s, n, caches) -> {"ctx", "att"}, {"S","V","cov"}
        raise NotImplementedError("No calling __call__ from Attention.")

    def prepare_cache(self, s):
        # prepare some pre-computed values to make it efficient
        raise NotImplementedError("No calling prepare_cache from Attention.")

    @staticmethod
    def get_attentioner(s):
        return {"ff":FfAttention, "biaff":BiaffAttention, "dummy":DmAttention}[s]

    # --- coverage related
    def has_cov(self):
        return self.n_cov > 0

    def init_cov(self, hid):
        # different hiddens for ff or biaffine
        self.params["cov2e"] = self._add_params((hid, self.n_cov))
        self.cov_gru = GruNode(self.model, [1, self.n_src, self.n_trg], self.n_cov)

    def output_cov(self, att_hidden, cov):
        return BK.affine([att_hidden, self.params["cov2e"], cov])

    def start_cov(self, ll, bsize):
        return BK.zeros((ll, self.n_cov), batch_size=bsize)

    def update_cov(self, att, s, h, cov):
        # for simplicity, no cached value for s and copy h (ignore since typically n_cov is not large)
        att = BK.transpose(att)
        # s = s
        h = BK.concatenate_cols([h for _ in range(BK.dims(att)[1])])
        return self.cov_gru([att, s, h], {"H": cov})["H"]
    # --- coverage related

    def refresh(self, argv):
        self._refresh(argv)
        if self.has_cov():
            self.cov_gru.refresh(argv)

class DmAttention(Attention):
    def __init__(self, model, n_src, n_trg, n_hidden, n_cov=0):
        super(DmAttention, self).__init__(model, n_src, n_trg, n_hidden, n_cov)
        self.n_hidden = None

    def __call__(self, s, n, caches):
        return {"ctx": BK.average(s), "att":None}, caches

# feed forward for attention --- requiring much memory
class FfAttention(Attention):
    def __init__(self, model, n_src, n_trg, n_hidden, n_cov=0):
        super(FfAttention, self).__init__(model, n_src, n_trg, n_hidden, n_cov)
        # parameters -- (feed-forward version)
        self.params["s2e"] = self._add_params((n_hidden, n_src))
        self.params["h2e"] = self._add_params((n_hidden, n_trg))
        self.params["be"] = self._add_params((n_hidden, ))
        self.params["e2a"] = self._add_params((1, n_hidden))
        if self.has_cov():
            self.init_cov(n_hidden)

    def prepare_cache(self, s):
        caches = {}
        caches["S"] = BK.concatenate_cols(s)
        caches["V"] = self.iparams["s2e"] * caches["S"]     # {(n_hidden, steps), batch_size}
        if self.has_cov():
            caches["cov"] = self.start_cov(len(s), BK.bsize(caches["V"]))
        return caches

    def __call__(self, s, n, caches):
        # s: list(len==steps) of {(n_s,), batch_size}, n: {(n_h,), batch_size}
        if caches is None:
            caches = self.prepare_cache(s)
        val_h = BK.affine(self.iparams["be"], self.iparams["h2e"], n)     # {(n_hidden,), batch_size}
        att_hidden_bef = BK.colwise_add(caches["V"], val_h)    # {(n_hidden, steps), batch_size}
        att_hidden = BK.tanh(att_hidden_bef)
        if self.has_cov():
            att_hidden = self.output_cov(att_hidden, caches["cov"])
        # if self.hdrop > 0:     # save some space
        #     att_hidden = BK.dropout(att_hidden, self.hdrop)
        att_e = BK.reshape(self.iparams["e2a"] * att_hidden, (len(s), ), batch_size=BK.bsize(att_hidden))
        att_alpha = BK.softmax(att_e)
        ctx = caches["S"] * att_alpha      # {(n_s, sent_len), batch_size}
        if self.has_cov():
            caches["cov"] = self.update_cov(att_alpha, caches["S"], n, caches["cov"])
        return {"ctx": ctx, "att": att_alpha}, caches

class BiaffAttention(Attention):
    def __init__(self, model, n_src, n_trg, n_hidden, n_cov=0):
        super(BiaffAttention, self).__init__(model, n_src, n_trg, n_hidden, n_cov)
        self.n_hidden = None
        # parameters -- (BiAffine-version e = h*W*s)
        self.params["W"] = self._add_params((n_trg, n_src))
        if self.has_cov():
            self.init_cov(n_trg)

    def prepare_cache(self, s):
        caches = {}
        caches["S"] = BK.concatenate_cols(s)
        caches["V"] = self.iparams["W"] * caches["S"]   # {(n_trg, steps), batch_size}
        if self.has_cov():
            caches["cov"] = self.start_cov(len(s), BK.bsize(caches["V"]))
        return caches

    def __call__(self, s, n, caches):
        # s: list(len==steps) of {(n_s,), batch_size}, n: {(n_h,), batch_size}
        if caches is None:
            caches = self.prepare_cache(s)
        wn_t = BK.transpose(n)
        att_hidden = caches["V"]
        if self.has_cov():
            att_hidden = self.output_cov(att_hidden, caches["cov"])
        att_e = BK.reshape(wn_t * att_hidden, (len(s), ), batch_size=BK.bsize(n))
        att_alpha = BK.softmax(att_e)
        ctx = caches["S"] * att_alpha
        if self.has_cov():
            caches["cov"] = self.update_cov(att_alpha, caches["S"], n, caches["cov"])
        return {"ctx": ctx, "att": att_alpha}, caches

# ================= Blocks ================= #
# stateless encoder
class Encoder(object):
    def __init__(self, model, n_input, n_hidden, n_layers, rnn_type):
        # [[f,b], ...]
        self.ntype = RnnNode.get_rnode(rnn_type)
        self.nodes = [[self.ntype(model, n_input, n_hidden), self.ntype(model, n_input, n_hidden)]]
        for i in range(n_layers-1):
            self.nodes.append([self.ntype(model, n_hidden, n_hidden), self.ntype(model, n_hidden, n_hidden)])
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers

    def refresh(self, argv):
        for nn in self.nodes:
            nn[0].refresh(argv)
            nn[1].refresh(argv)

    def __call__(self, embeds, masks):
        # todo(warn), only put masks here in enc
        # embeds: list(step) of {(n_emb, ), batch_size}, using padding for batches
        b_size = BK.bsize(embeds[0])
        outputs = [embeds]
        for i, nn in zip(range(self.n_layers), self.nodes):
            init_hidden = BK.zeros((self.n_hidden,), batch_size=b_size)
            tmp_f = []      # forward
            tmp_f_prev = {"H":init_hidden, "C":init_hidden}
            for e, m in zip(outputs[-1], masks):
                one_output = nn[0](e, tmp_f_prev, m)
                tmp_f.append(one_output["H"])
                tmp_f_prev = one_output
            tmp_b = []      # forward
            tmp_b_prev = {"H":init_hidden, "C":init_hidden}
            for e, m in zip(reversed(outputs[-1]), reversed(masks)):
                one_output = nn[1](e, tmp_b_prev, m)
                tmp_b.append(one_output["H"])
                tmp_b_prev = one_output
            # concat
            ctx = [BK.concatenate([f,b]) for f,b in zip(tmp_f, reversed(tmp_b))]
            outputs.append(ctx)
        return outputs[-1]

# -------------
# stateless attentional decoder
class Decoder(object):
    def __init__(self, model, n_inputs, n_hidden, n_layers, rnn_type, summ_type, att_type, att_n_hidden, att_cov_n, dim_src):
        self.ntype = RnnNode.get_rnode(rnn_type)
        self.all_nodes = []
        # gru nodes --- wait for the sub-classes
        # init nodes
        self.inodes = [Affine(model, dim_src, n_hidden, act="tanh") for _ in range(n_layers)]
        for inod in self.inodes:
            self.all_nodes.append(inod)
        # att node
        self.anode = Attention.get_attentioner(att_type)(model, dim_src, n_hidden, att_n_hidden, att_cov_n)
        self.all_nodes.append(self.anode)
        # info
        if not isinstance(n_inputs, Iterable):
            n_inputs = [n_inputs]
        self.n_input = n_inputs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dim_src = dim_src      # also the size of attention vector
        # summarize for the source as the start of decoder
        self.summer = Decoder.get_summer(summ_type, dim_src)    # bidirection

    @staticmethod
    def get_summer(s, size):  # list of values (bidirection) => one value
        if s == "avg":
            return BK.average
        else:
            mask = [0. for _ in range(size//2)]+[1. for _ in range(size//2)]
            mask2 = [1. for _ in range(size//2)]+[0. for _ in range(size//2)]
            if s == "fend":
                return lambda x: BK.cmult(BK.inputVector(mask2), x[-1])
            elif s == "bend":
                return lambda x: BK.cmult(BK.inputVector(mask), x[0])
            elif s == "ends":
                return lambda x: BK.cmult(BK.inputVector(mask2), x[-1]) + BK.cmult(BK.inputVector(mask), x[0])
            else:
                return None

    def refresh(self, argv):
        for nn in self.all_nodes:
            nn.refresh(argv)    # dropouts: init/att?: hdrop, rec: idrop, gdrop

    def start_one(self, ss):
        # ss: list of srcs (to be attended), return
        inits = []
        summ = self.summer(ss)
        for i in range(self.n_layers):
            cur_init = self.inodes[i](summ)
            # +1 for the init state
            inits.append({"H": cur_init, "C": BK.zeros((self.n_hidden,), batch_size=BK.bsize(cur_init))})
        att_res, att_caches = self.anode(ss, inits[0]["H"], None)          # start of the attention
        att_res["hidden"] = inits
        return att_res, att_caches

    def feed_one(self, ss, inputs, hiddens, att_caches):
        # input ones
        if not isinstance(inputs, Iterable):
            inputs = [inputs]
        # check batch-size, todo
        # caches are only for attentions
        return self._feed_one(ss, inputs, hiddens, att_caches)

    def _feed_one(self, ss, inputs, hiddens, caches):
        # [src], [inputs], [hiddens], caches -> {ctx, att, hid}, {<ATT-caches>}
        raise NotImplementedError("Decoder should be inherited!")

# normal attention decoder
class AttDecoder(Decoder):
    def __init__(self, model, n_inputs, n_hidden, n_layers, rnn_type, summ_type, att_type, att_n_hidden, att_cov_n, dim_src):
        super(AttDecoder, self).__init__(model, n_inputs, n_hidden, n_layers, rnn_type, summ_type, att_type, att_n_hidden, att_cov_n, dim_src)
        # gru nodes
        self.gnodes = [self.ntype(model, n_inputs+[dim_src], n_hidden)]    # (E(y_{i-1})//c_i, s_{i-1}) => s_i
        for i in range(n_layers-1):
            self.gnodes.append(self.ntype(model, n_hidden, n_hidden))
        for gnod in self.gnodes:
            self.all_nodes.append(gnod)

    def _feed_one(self, ss, inputs, hiddens, att_caches):
        # first layer with attetion
        att_res, att_caches = self.anode(ss, hiddens[0]["H"], att_caches)
        hidd = self.gnodes[0](inputs+[att_res["result"]], hiddens[0])
        this_hiddens = [hidd]
        # later layers
        for i in range(1, self.n_layers):
            ihidd = self.gnodes[i](this_hiddens[i-1]["H"], hiddens[i])
            this_hiddens.append(ihidd)
        return utils.Helper.combine_dicts(att_res, {"hid": this_hiddens}), att_caches

# nematus-style attention decoder, fixed two transitions
class NematusDecoder(Decoder):
    def __init__(self, model, n_inputs, n_hidden, n_layers, rnn_type, summ_type, att_type, att_n_hidden, att_cov_n, dim_src):
        super(NematusDecoder, self).__init__(model, n_inputs, n_hidden, n_layers, rnn_type, summ_type, att_type, att_n_hidden, att_cov_n, dim_src)
        # gru nodes
        self.gnodes = [self.ntype(model, n_inputs, n_hidden)]        # gru1 for the first layer
        for i in range(n_layers-1):
            self.gnodes.append(self.ntype(model, n_hidden, n_hidden))
        self.gnodes.append(self.ntype(model, dim_src, n_hidden))   # gru2 for the first layer
        for gnod in self.gnodes:
            self.all_nodes.append(gnod)

    def _feed_one(self, ss, inputs, hiddens, att_caches):
        # first layer with attetion, gru1 -> att -> gru2
        s1 = self.gnodes[0](inputs, hiddens[0])
        att_res, att_caches = self.anode(ss, s1["H"], att_caches)
        hidd = self.gnodes[-1](att_res["result"], s1)
        this_hiddens = [hidd]
        # later layers
        for i in range(1, self.n_layers):
            ihidd = self.gnodes[i](this_hiddens[i-1]["H"], hiddens[i])
            this_hiddens.append(ihidd)
        return utils.Helper.combine_dicts(att_res, {"hid": this_hiddens}), att_caches
