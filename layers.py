# layers for nn

import dynet as dy
import numpy as np

# ================= Helpers ====================== #
# get mask inputs
def gen_masks_input(rate, size, bsize=1):
    def _gen_masks(size, rate):
        # inverted dropout
        x = np.random.binomial(1, rate, size).astype(np.float)
        x *= (1.0/rate)
        return x
    x = _gen_masks((size, bsize), rate)
    return dy.inputTensor(x, True)

def gen_maks_embed(rate, num):
    # hope this might not be too costy
    # also shared dropping in one minibatch for convenience
    x = set()
    if type(num) == int:
        rr = np.random.binomial(1, rate, num)
        for i, n in enumerate(rr):
            x.add(i) if n>0 else None
    else:
        rr = np.random.binomial(1, rate, len(num))
        for i, n in zip(num, rr):
            x.add(i) if n>0 else None
    return x

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
        x = self.iparams["W"]*input_exp+self.iparams["B"]
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
class GruNode(Basic):
    def __init__(self, model, n_input, n_hidden):
        super(GruNode, self).__init__(model)
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
    def __init__(self, model, n_s, n_h, n_hidden):
        super(Attention, self).__init__(model)
        # parameters -- (feed-forward version)
        self.params["s2e"] = self._add_parameters((n_hidden, n_s))
        self.params["h2e"] = self._add_parameters((n_hidden, n_h))
        self.params["v"] = self._add_parameters((1, n_hidden))
        # cache value
        self.cache_sig = None
        self.cache_v = None
        self.cache_concat = None

    def refresh(self, **argv):
        # additionally clear cache
        self.cache_concat = None
        self.cache_sig = None
        self.cache_v = None     # {(n_hidden, sent_len), batch_size}
        # refresh (no masks here)
        self.drop = float(argv["hdrop"]) if "hdrop" in argv else 0.
        self._ingraph(argv)

    def __call__(self, s, n):
        # s: list(len==steps) of {(n_s,), batch_size}, n: {(n_h,), batch_size}
        # if True:    # debug
        #     return dy.average(s)
        sig = str(len(s))+str(s[0])
        if sig != self.cache_sig:
            # calculate for the results of s2e*s
            self.cache_sig = sig
            self.cache_concat = dy.concatenate_cols(s)
            self.cache_v = self.iparams["s2e"] * self.cache_concat
        val_h = self.iparams["h2e"] * n     # {(n_hidden,), batch_size}
        att_hidden_bef = dy.colwise_add(self.cache_v, val_h)    # {(n_didden, steps), batch_size}
        att_hidden = dy.tanh(att_hidden_bef)
        # if self.drop > 0:     # save some space
        #     att_hidden = dy.dropout(att_hidden, self.drop)
        att_e = dy.reshape(self.iparams["v"] * att_hidden, (len(s), ), batch_size=att_hidden.dim()[1])
        att_alpha = dy.softmax(att_e)
        ctx = self.cache_concat * att_alpha      # {(n_s, sent_len), batch_size}
        return ctx

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
        b_size = embeds[0].dim()[1]
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

# attentional decoder
class Decoder(object):
    def __init__(self, model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden):
        self.ntype = GruNode
        self.all_nodes = []
        # gru nodes --- wait for the sub-classes
        # init nodes
        self.inodes = [Linear(model, dim_src, n_hidden, act="tanh") for _ in range(n_layers)]
        for inod in self.inodes:
            self.all_nodes.append(inod)
        # att node
        self.anode = Attention(model, dim_src, n_hidden, dim_att_hidden)
        self.all_nodes.append(self.anode)
        # caches
        self.s = None
        self.cache_att = []
        self.cache_hiddens = [[] for _ in range(n_layers)]  # +1 for the init state
        # info
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dim_src = dim_src      # also the size of attention vector

    def refresh(self, **argv):
        self.s = None
        self.cache_att = []
        self.cache_hiddens = [[] for _ in range(self.n_layers)]
        for nn in self.all_nodes:
            nn.refresh(**argv)    # dropouts: init/att: hdrop, rec: idrop, gdrop

    def start_one(self, s):
        # start to decode with one sentence, W*avg(s) as init
        self.s = s
        avg = dy.average(s)
        for i in range(self.n_layers):
            cur_init = self.inodes[i](avg)
            self.cache_hiddens[i].append(cur_init)          # +1 for the init state

    def feed_one(self, inputs):
        assert self.started, "Decoder has not been started!!"
        # one or several steps forward, return the last states
        self._feed_one(inputs)
        return self.cache_att[-1], self.cache_hiddens[-1][-1]

    def _feed_one(self, inputs):
        raise NotImplementedError("Decoder should be inherited!")

    @property
    def started(self):
        return self.s is not None

    def get_results(self):
        # return att and last-layer hiddens
        return self.cache_att, self.cache_hiddens[-1][1:]   # +1 for the init state

# normal attention decoder
class AttDecoder(Decoder):
    def __init__(self, model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden):
        super(AttDecoder, self).__init__(model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden)
        # gru nodes
        self.gnodes = [self.ntype(model, n_input+dim_src, n_hidden)]    # (E(y_{i-1})//c_i, s_{i-1}) => s_i
        for i in range(n_layers-1):
            self.gnodes.append(self.ntype(model, n_hidden, n_hidden))
        for gnod in self.gnodes:
            self.all_nodes.append(gnod)

    def _feed_one(self, inputs):
        # input ones
        if type(inputs) != list:
            inputs = [inputs]
        # feed one at a time
        for one in inputs:
            # first layer with attetion
            att = self.anode(self.s, self.cache_hiddens[0][-1])
            self.cache_att.append(att)
            g_input = dy.concatenate([one, att])
            hidd = self.gnodes[0](g_input, self.cache_hiddens[0][-1])
            self.cache_hiddens[0].append(hidd)
            # later layers
            for i in range(1, self.n_layers):
                ihidd = self.gnodes[i](self.cache_hiddens[i-1][-1], self.cache_hiddens[i][-1])
                self.cache_hiddens[i].append(ihidd)

# nematus-style attention decoder, fixed two transitions
class NematusDecoder(Decoder):
    def __init__(self, model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden):
        super(NematusDecoder, self).__init__(model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden)
        # gru nodes
        self.gnodes = [self.ntype(model, n_input, n_hidden)]        # gru1 for the first layer
        for i in range(n_layers-1):
            self.gnodes.append(self.ntype(model, n_hidden, n_hidden))
        self.gnodes.append(self.ntype(model, dim_src, n_hidden))   # gru2 for the first layer
        for gnod in self.gnodes:
            self.all_nodes.append(gnod)

    def _feed_one(self, inputs):
        # input ones
        if type(inputs) != list:
            inputs = [inputs]
        # feed one at a time
        for one in inputs:
            # first layer with attetion, gru1 -> att -> gru2
            s1 = self.gnodes[0](one, self.cache_hiddens[0][-1])
            att = self.anode(self.s, s1)
            self.cache_att.append(att)
            hidd = self.gnodes[-1](att, s1)
            self.cache_hiddens[0].append(hidd)
            # later layers
            for i in range(1, self.n_layers):
                ihidd = self.gnodes[i](self.cache_hiddens[i-1][-1], self.cache_hiddens[i][-1])
                self.cache_hiddens[i].append(ihidd)
