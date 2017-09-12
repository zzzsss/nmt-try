# layers for nn

import dynet as dy
import numpy as np

# ================= Helpers ====================== #
# get mask inputs
def gen_masks_input(rate, size, bsize=1):
    def _gen_masks(size, rate):
        # inverted dropout
        r = 1-rate
        x = np.random.binomial(1, r, size).astype(np.float)
        x *= (1.0/r)
        return x
    x = _gen_masks((size, bsize), rate)
    return dy.inputTensor(x, True)

def gen_maks_embed(rate, num):
    # hope this might not be too costy
    # also shared dropping in one minibatch for convenience
    x = set()
    r = 1-rate
    if type(num) == int:
        rr = np.random.binomial(1, r, num)
        for i, n in enumerate(rr):
            x.add(i) if n>0 else None
    else:
        rr = np.random.binomial(1, r, len(num))
        for i, n in zip(num, rr):
            x.add(i) if n>0 else None
    return x

def bs(x):
    return x.dim()[1]

# ================= Basic Blocks ================= #
# basic unit (stateful about dropouts)
class Basic(object):
    def __init__(self, model):
        self.model = model
        self.params = {}
        self.iparams = {}
        self.update = None
        # dropouts
        self.drop = 0.
        self.masks = None

    def _ingraph(self, argv):
        # update means whether the parameters should be updated
        update = bool(argv["update"]) if "update" in argv else True
        ingraph = bool(argv["ingraph"]) if "ingraph" in argv else True
        if ingraph:
            for k in self.params:
                self.iparams[k] = dy.parameter(self.params[k], update)
            self.update = update

    def refresh(self):
        # argvs include: hdrop=0., idrop=0., gdrop=0., update=True, ingraph=True
        raise NotImplementedError("No calling refresh from Basic.")

    def _add_parameters(self, shape, lookup=False):
        def ortho_weight(ndim):
            W = np.random.randn(ndim, ndim)
            u, s, v = np.linalg.svd(W)
            return u.astype(np.float)
        if lookup:
            return self.model.add_lookup_parameters(shape)  # also default Glorot
        # shape is a tuple of dims
        if len(shape) == 1:     # set bias to 0
            return self.model.add_parameters(shape, init=dy.ConstInitializer(0.))
        else:
            if len(shape)==2 and shape[0]==shape[1]:    # not exact criterion for rec-param, but might be ok
                arr = ortho_weight(shape[0])
                return self.model.add_parameters(shape, init=dy.NumpyInitializer(arr))
            else:
                return self.model.add_parameters(shape, init=dy.GlorotInitializer())

# linear layer with selectable activation functions
class Linear(Basic):
    def __init__(self, model, n_in, n_out, act="tanh"):
        super(Linear, self).__init__(model)
        self.params["W"] = self._add_parameters((n_out, n_in))
        self.params["B"] = self._add_parameters((n_out,))
        self.act = {"tanh":dy.tanh, "softmax":dy.softmax, "linear":None}[act]

    def refresh(self, **argv):
        self.drop = float(argv["hdrop"]) if "hdrop" in argv else 0.
        self._ingraph(argv)

    def __call__(self, input_exp):
        x = dy.affine_transform([self.iparams["B"], self.iparams["W"], input_exp])
        if self.act is not None:
            x = self.act(x)
        if self.drop > 0.:
            x = dy.dropout(x, self.drop)
        return x

# embedding layer
class Embedding(Basic):
    def __init__(self, model, n_words, n_dim, dropout_wordceil=None, npvec=None):
        super(Embedding, self).__init__(model)
        if npvec is not None:
            assert len(npvec.shape) == 2 and npvec.shape[0] == n_words and npvec.shape[1] == n_dim
            self.params["E"] = self.model.lookup_parameters_from_numpy(npvec)
        else:
            self.params["E"] = self._add_parameters((n_words, n_dim), lookup=True)
        self.n_dim = n_dim
        self.n_words = n_words
        self.dropout_wordceil = dropout_wordceil if dropout_wordceil is not None else n_words

    def refresh(self, **argv):
        # zero out
        self.params["E"].init_row(0, [0. for _ in range(self.n_dim)])
        # refresh
        self.drop = float(argv["hdrop"]) if "hdrop" in argv else 0.
        self._ingraph(argv)
        self.masks = None
        gdrop = float(argv["gdrop"]) if "gdrop" in argv else 0.
        words = argv["words"] if "words" in argv else self.dropout_wordceil     # list or ceiling-num
        if gdrop > 0:
            self.masks = gen_maks_embed(gdrop, words)

    def __call__(self, input_exp):
        # input should be a list of ints, masks should be a set of ints for the dropped ints
        # input dropout
        if self.masks is not None:
            if type(input_exp) == list:
                input_exp = [(i if (i not in self.masks) else 0) for i in input_exp]
            elif input_exp in self.masks:
                input_exp = 0
        if type(input_exp) == list:
            x = dy.lookup_batch(self.params["E"], input_exp, self.update)
        else:
            x = dy.lookup(self.params["E"], input_exp, self.update)
        if self.drop > 0:
            x = dy.dropout(x, self.drop)
        return x

# rnn nodes
class RnnNode(Basic):
    def __init__(self, model, n_input, n_hidden):
        super(RnnNode, self).__init__(model)
        self.masks = (None, None)
        self.n_input = n_input
        self.n_hidden = n_hidden

    def refresh(self, **argv):
        # refresh
        self.drop = float(argv["idrop"]) if "idrop" in argv else 0.
        self._ingraph(argv)
        self.masks = (None, None)
        gdrop = float(argv["gdrop"]) if "gdrop" in argv else 0.
        bsize = int(argv["bsize"]) if "bsize" in argv else 1       # if not, the same mask for all elements in batch
        if gdrop > 0:
            self.masks = (gen_masks_input(gdrop, self.n_input, bsize), gen_masks_input(gdrop, self.n_hidden, bsize))
        # TODO ?? don't remember what this todo is to do

    def __call__(self, input_exp, hidden_exp):
        raise NotImplementedError()

    @staticmethod
    def get_rnode(s):  #todo: lstm
        return {"gru":GruNode, "lstm":None, "dummy":DmRnnNode}[s]

class DmRnnNode(RnnNode):
    def __init__(self, model, n_input, n_hidden):
        super(DmRnnNode, self).__init__(model, n_input, n_hidden)

    def __call__(self, input_exp, hidden_exp):
        return hidden_exp

class GruNode(RnnNode):
    def __init__(self, model, n_input, n_hidden):
        super(GruNode, self).__init__(model, n_input, n_hidden)
        # paramters
        self.params["x2r"] = self._add_parameters((n_hidden, n_input))
        self.params["h2r"] = self._add_parameters((n_hidden, n_hidden))
        self.params["br"] = self._add_parameters((n_hidden,))
        self.params["x2z"] = self._add_parameters((n_hidden, n_input))
        self.params["h2z"] = self._add_parameters((n_hidden, n_hidden))
        self.params["bz"] = self._add_parameters((n_hidden,))
        self.params["x2h"] = self._add_parameters((n_hidden, n_input))
        self.params["h2h"] = self._add_parameters((n_hidden, n_hidden))
        self.params["bh"] = self._add_parameters((n_hidden,))

    def __call__(self, input_exp, hidden_exp):
        # two kinds of dropouts
        if self.masks[0] is not None:
            input_exp = dy.cmult(self.masks[0], input_exp)
        if self.masks[1] is not None:
            hidden_exp = dy.cmult(self.masks[1], hidden_exp)
        if self.drop > 0.:
            input_exp = dy.dropout(input_exp, self.drop)
        rt = dy.affine_transform([self.iparams["br"], self.iparams["x2r"], input_exp, self.iparams["h2r"], hidden_exp])
        rt = dy.logistic(rt)
        zt = dy.affine_transform([self.iparams["bz"], self.iparams["x2z"], input_exp, self.iparams["h2z"], hidden_exp])
        zt = dy.logistic(zt)
        h_reset = dy.cmult(rt, hidden_exp)
        ht = dy.affine_transform([self.iparams["bh"], self.iparams["x2h"], input_exp, self.iparams["h2h"], h_reset])
        ht = dy.tanh(ht)
        hidden = dy.cmult(zt, hidden_exp) + dy.cmult((1. - zt), ht)
        return hidden

class Attention(Basic):
    def __init__(self, model, n_s, n_h):
        super(Attention, self).__init__(model)
        # cache value
        self.get_sig = lambda s: str(len(s))+str(s[0])
        self.cache_sig = None
        self.cache_values = {}
        self.need_rerange = {"S":True, "V":True}
        # info
        self.n_s, self.n_h = n_s, n_h

    def _refresh(self, **argv):
        self.cache_values = {}
        # self.drop = float(argv["hdrop"]) if "hdrop" in argv else 0.
        self._ingraph(argv)

    def refresh(self, **argv):
        raise NotImplementedError("No calling refresh from Attention.")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("No calling __call__ from Attention.")

    # todo(warn): dangerous !!
    def rerange_cache(self, orders):
        for k in self.cache_values:
            self.cache_values[k] = dy.pick_batch_elems(self.cache_values[k], orders)

    @staticmethod
    def get_attentioner(s):
        return {"ff":FfAttention, "biaff":BiaffAttention, "dummy":DmAttention}[s]

class DmAttention(Attention):
    def __init__(self, model, n_s, n_h, n_hidden):
        super(DmAttention, self).__init__(model, n_s, n_h)

    def refresh(self, **argv):
        super(DmAttention, self)._refresh(**argv)

    def __call__(self, s, n):
        return dy.average(s), None  # todo(warn) do not use for decoding

# feed forward for attention --- requiring much memory
class FfAttention(Attention):
    def __init__(self, model, n_s, n_h, n_hidden):
        super(FfAttention, self).__init__(model, n_s, n_h)
        # parameters -- (feed-forward version)
        self.params["s2e"] = self._add_parameters((n_hidden, n_s))
        self.params["h2e"] = self._add_parameters((n_hidden, n_h))
        self.params["v"] = self._add_parameters((1, n_hidden))

    def refresh(self, **argv):
        super(FfAttention, self)._refresh(**argv)

    def __call__(self, s, n):
        # s: list(len==steps) of {(n_s,), batch_size}, n: {(n_h,), batch_size}
        sig = self.get_sig(s)
        # calculate for the results of caches if not present
        if sig != self.cache_sig:
            self.cache_sig = sig
            self.cache_values["S"] = dy.concatenate_cols(s)
            self.cache_values["V"] = self.iparams["s2e"] * self.cache_values["S"]
        val_h = self.iparams["h2e"] * n     # {(n_hidden,), batch_size}
        att_hidden_bef = dy.colwise_add(self.cache_values["V"], val_h)    # {(n_didden, steps), batch_size}
        att_hidden = dy.tanh(att_hidden_bef)
        # if self.drop > 0:     # save some space
        #     att_hidden = dy.dropout(att_hidden, self.drop)
        att_e = dy.reshape(self.iparams["v"] * att_hidden, (len(s), ), batch_size=bs(att_hidden))
        att_alpha = dy.softmax(att_e)
        ctx = self.cache_values["S"] * att_alpha      # {(n_s, sent_len), batch_size}
        # if True:    # debug (with dev(bs80, h1000): 4186->bef->8057->tanh->10491->dropout->14138, without avg:11033)
        #     return dy.average(s)
        return ctx, att_alpha

class BiaffAttention(Attention):
    def __init__(self, model, n_s, n_h, n_hidden):
        super(BiaffAttention, self).__init__(model, n_s, n_h)
        # parameters -- (BiAffine-version e = h*W*s)
        self.params["W"] = self._add_parameters((n_s, n_h))

    def refresh(self, **argv):
        super(BiaffAttention, self)._refresh(**argv)

    def __call__(self, s, n):
        # s: list(len==steps) of {(n_s,), batch_size}, n: {(n_h,), batch_size}
        sig = self.get_sig(s)
        if sig != self.cache_sig:
            self.cache_sig = sig
            self.cache_values["S"] = dy.concatenate_cols(s)
        wn = self.iparams["W"] * n      # {(n_s,), batch_size}
        wn_t = dy.reshape(wn, (1, self.n_s), batch_size=bs(n))
        att_e = dy.reshape(wn_t * self.cache_values["S"], (len(s), ), batch_size=bs(n))
        att_alpha = dy.softmax(att_e)
        ctx = self.cache_values["S"] * att_alpha
        return ctx, att_alpha

# ================= Blocks ================= #
# stateless encoder
class Encoder(object):
    def __init__(self, model, n_input, n_hidden, n_layers):
        # [[f,b], ...]
        self.ntype = GruNode
        self.nodes = [[self.ntype(model, n_input, n_hidden), self.ntype(model, n_input, n_hidden)]]
        for i in range(n_layers-1):
            self.nodes.append([self.ntype(model, n_hidden, n_hidden), self.ntype(model, n_hidden, n_hidden)])
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers

    def refresh(self, **argv):
        for nn in self.nodes:
            nn[0].refresh(**argv)
            nn[1].refresh(**argv)

    def __call__(self, embeds):
        # embeds: list(step) of {(n_emb, ), batch_size}, using padding for batches
        b_size = bs(embeds[0])
        outputs = [embeds]
        for i, nn in zip(range(self.n_layers), self.nodes):
            init_hidden = dy.zeroes((self.n_hidden,), batch_size=b_size)
            tmp_f = []      # forward
            tmp_f_prev = init_hidden
            for e in outputs[-1]:
                one_output = nn[0](e, tmp_f_prev)
                tmp_f.append(one_output)
                tmp_f_prev = one_output
            tmp_b = []      # forward
            tmp_b_prev = init_hidden
            for e in reversed(outputs[-1]):
                one_output = nn[1](e, tmp_b_prev)
                tmp_b.append(one_output)
                tmp_b_prev = one_output
            # concat
            ctx = [dy.concatenate([f,b]) for f,b in zip(tmp_f, reversed(tmp_b))]
            outputs.append(ctx)
        return outputs[-1]

# -------------
class DecoderState(object):
    def __init__(self, dec, s, n_layers, hiddens, att, attw, prev=None, attender=None):
        self.dec = dec
        self.n_layers = n_layers
        # caches
        self.s = s
        self.cache_hiddens = hiddens
        self.cache_att = att
        self.cache_attw = attw      # attention weights
        self.prev = prev
        self.attender = prev.attender if attender is None else attender

    @property
    def bsize(self):
        return bs(self.cache_hiddens[-1])

    # todo(warn) this one is quite dangerous, use carefully !!
    # basically it handles the reranging of state-hiddens and attention-hiddens
    def rerange_cache(self, orders, att_orders):
        if orders is not None:
            self.cache_hiddens = [dy.pick_batch_elems(one, orders) for one in self.cache_hiddens]
            self.cache_att = dy.pick_batch_elems(self.cache_att, orders)
            self.cache_attw = None      # just clear that
            if att_orders is not None:
                self.attender.rerange_cache(att_orders)

    # !! should not be used after any shuffling
    def get_results(self):
        if self.prev is None:
            return [self.cache_hiddens[-1]], [self.cache_att], [self.cache_attw]
        hiddens, atts, attws = self.prev.get_results()
        hiddens.append(self.cache_hiddens[-1])  # last layer
        atts.append(self.cache_att)
        return hiddens, atts, attws

    def get_results_one(self):
        return self.cache_hiddens[-1], self.cache_att, self.cache_attw

# attentional decoder
class Decoder(object):
    def __init__(self, model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden, att_type, rnn_type):
        self.ntype = RnnNode.get_rnode(rnn_type)
        self.all_nodes = []
        # gru nodes --- wait for the sub-classes
        # init nodes
        self.inodes = [Linear(model, dim_src, n_hidden, act="tanh") for _ in range(n_layers)]
        for inod in self.inodes:
            self.all_nodes.append(inod)
        # att node
        self.anode = Attention.get_attentioner(att_type)(model, dim_src, n_hidden, dim_att_hidden)
        self.all_nodes.append(self.anode)
        # info
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dim_src = dim_src      # also the size of attention vector

    def refresh(self, **argv):
        for nn in self.all_nodes:
            nn.refresh(**argv)    # dropouts: init/att: hdrop, rec: idrop, gdrop

    def start_one(self, s, expand=1):
        # start to decode with one sentence, W*avg(s) as init
        # also expand on the batch dimension # TODO: maybe should put at the col dimension
        inits = []
        avg = dy.average(s)
        if expand > 1:      # waste of memory, but seems to be a feasible way
            indices = []
            for i in range(bs(avg)):
                indices += [i for _ in range(expand)]
            avg = dy.pick_batch_elems(avg, indices)
            s = [dy.pick_batch_elems(one, indices) for one in s]
        for i in range(self.n_layers):
            cur_init = self.inodes[i](avg)
            inits.append(cur_init)          # +1 for the init state
        att, attw = self.anode(s, inits[0])          # start of the attention
        return DecoderState(self, s, self.n_layers, inits, att, attw, attender=self.anode)

    def feed_one(self, ss, inputs):
        # one or several steps forward, return the last states
        # input ones
        if type(inputs) != list:
            inputs = [inputs]
        # check batch-size
        assert all([ss.bsize == bs(i) for i in inputs]), "Unmatched batch_size"
        # feed one at a time
        for one in inputs:
            ss = self._feed_one(ss, one)
        return ss

    def _feed_one(self, ss, one):
        raise NotImplementedError("Decoder should be inherited!")

# normal attention decoder
class AttDecoder(Decoder):
    def __init__(self, model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden, att_type, rnn_type):
        super(AttDecoder, self).__init__(model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden, att_type, rnn_type)
        # gru nodes
        self.gnodes = [self.ntype(model, n_input+dim_src, n_hidden)]    # (E(y_{i-1})//c_i, s_{i-1}) => s_i
        for i in range(n_layers-1):
            self.gnodes.append(self.ntype(model, n_hidden, n_hidden))
        for gnod in self.gnodes:
            self.all_nodes.append(gnod)

    def _feed_one(self, ss, one):
        # first layer with attetion
        att, attw = self.anode(ss.s, ss.cache_hiddens[0])
        g_input = dy.concatenate([one, att])
        hidd = self.gnodes[0](g_input, ss.cache_hiddens[0])
        this_hiddens = [hidd]
        # later layers
        for i in range(1, self.n_layers):
            ihidd = self.gnodes[i](this_hiddens[i-1], ss.cache_hiddens[i])
            this_hiddens.append(ihidd)
        return DecoderState(self, ss.s, ss.n_layers, this_hiddens, att, attw, prev=ss)

# nematus-style attention decoder, fixed two transitions
class NematusDecoder(Decoder):
    def __init__(self, model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden, att_type, rnn_type):
        super(NematusDecoder, self).__init__(model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden, att_type, rnn_type)
        # gru nodes
        self.gnodes = [self.ntype(model, n_input, n_hidden)]        # gru1 for the first layer
        for i in range(n_layers-1):
            self.gnodes.append(self.ntype(model, n_hidden, n_hidden))
        self.gnodes.append(self.ntype(model, dim_src, n_hidden))   # gru2 for the first layer
        for gnod in self.gnodes:
            self.all_nodes.append(gnod)

    def _feed_one(self, ss, one):
        # first layer with attetion, gru1 -> att -> gru2
        s1 = self.gnodes[0](one, ss.cache_hiddens[0])
        att, attw = self.anode(ss.s, s1)
        hidd = self.gnodes[-1](att, s1)
        this_hiddens = [hidd]
        # later layers
        for i in range(1, self.n_layers):
            ihidd = self.gnodes[i](this_hiddens[i-1], ss.cache_hiddens[i])
            this_hiddens.append(ihidd)
        return DecoderState(self, ss.s, ss.n_layers, this_hiddens, att, attw, prev=ss)
