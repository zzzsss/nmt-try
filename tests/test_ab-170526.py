import numpy as np
import sys, time, random, json
import dynet as dy

random.seed(12345)
np.random.seed(12345)

class Basic(object):
    def __init__(self, model):
        self.pc = model
        self.params = {}
        self.iparams = {}

    def ingraph(self, update=True):
        for k in self.params:
            self.iparams[k] = dy.parameter(self.params[k], update)

class GruNode(Basic):
    def __init__(self, model, n_input, n_hidden):
        super(GruNode, self).__init__(model)
        # paramters
        self.params["x2r"] = self.pc.add_parameters((n_hidden, n_input))
        self.params["h2r"] = self.pc.add_parameters((n_hidden, n_hidden))
        self.params["br"] = self.pc.add_parameters((n_hidden,), init=dy.ConstInitializer(0.))
        self.params["x2z"] = self.pc.add_parameters((n_hidden, n_input))
        self.params["h2z"] = self.pc.add_parameters((n_hidden, n_hidden))
        self.params["bz"] = self.pc.add_parameters((n_hidden,), init=dy.ConstInitializer(0.))
        self.params["x2h"] = self.pc.add_parameters((n_hidden, n_input))
        self.params["h2h"] = self.pc.add_parameters((n_hidden, n_hidden))
        self.params["bh"] = self.pc.add_parameters((n_hidden,), init=dy.ConstInitializer(0.))
        self.spec = n_input, n_hidden

    def __call__(self, input_exp, hidden_exp):
        rt = dy.affine_transform([self.iparams["br"], self.iparams["x2r"], input_exp, self.iparams["h2r"], hidden_exp])
        rt = dy.logistic(rt)
        zt = dy.affine_transform([self.iparams["bz"], self.iparams["x2z"], input_exp, self.iparams["h2z"], hidden_exp])
        zt = dy.logistic(zt)
        h_reset = dy.cmult(rt, hidden_exp)
        ht = dy.affine_transform([self.iparams["bh"], self.iparams["x2h"], input_exp, self.iparams["h2h"], h_reset])
        ht = dy.tanh(ht)
        hidden = dy.cmult(zt, hidden_exp) + dy.cmult((1. - zt), ht)
        return hidden


N_WORDS = 100
N_HIDDEN = 500
N_EMBED = 100
N_BATCH = 128
N_BATCHES_ITER = 100
N_BEAM = 16
N_STEP = 50
N_ITER = 3

for s in sys.argv:
    fs = s.split("=")
    if len(fs)==2 and str.upper(fs[0])==fs[0] and fs[0] in globals():
        globals()[fs[0]] = fs[1]
x = {}
tmp_keys = [x for x in globals().keys() if str.upper(x)==x]
print("Globals: %s" % ["[%s=%s]"%(x, globals()[x]) for x in tmp_keys])

m = dy.Model()
lp = m.add_lookup_parameters((N_WORDS, N_EMBED))
ss = m.add_parameters((N_WORDS, N_HIDDEN))
gru = GruNode(m, N_EMBED, N_HIDDEN)


# helpers #
class Timer:
    NAMED = {}
    def __init__(self, name=None, cname=None, print_date=False, quiet=False, info=""):
        self.name = name
        self.cname = cname
        self.print_date = print_date
        self.quiet = quiet
        self.info = info
        self.accu = 0.   # accumulated time
        self.paused = False
        self.start = time.time()

    def pause(self):
        if not self.paused:
            cur = time.time()
            self.accu += cur - self.start
            self.start = cur
            self.paused = True

    def resume(self):
        if not self.paused:
            printing("--- Timer should be paused to be resumed.", func="warn")
        else:
            self.start = time.time()
            self.paused = False

    def end(self):
        self.pause()
        if self.cname is not None:
            if self.cname not in Timer.NAMED:
                Timer.NAMED[self.cname] = 0
            Timer.NAMED[self.cname] += self.accu

    def __enter__(self):
        printing("-- Start timer %s: %s at %s." % (self.name, self.info, time.time())) if not self.quiet else None
        print_date() if self.print_date and not self.quiet else None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
        print_date() if self.print_date and not self.quiet else None
        printing("-- End timer %s at %s." % (self.name, time.time())) if not self.quiet else None
        return False

    @staticmethod
    def show_accu(x=""):
        if x == "":
            print(Timer.NAMED)
        else:
            print("%s: %s"%(x,Timer.NAMED[x]))

    @staticmethod
    def del_accu(x=""):
        if x == "":
            Timer.NAMED = {}
        else:
            Timer.NAMED[x] = 0.

class Accu:
    NAMED = {}
    METHODS = {}
    INITS = {}
    @staticmethod
    def show_nums(x=""):
        if x == "":
            print(Accu.NAMED)
        else:
            print("%s: %s"%(x,Accu.NAMED[x]))

    @staticmethod
    def del_nums(x=""):
        if x == "":
            Accu.NAMED = {}
        else:
            Accu.NAMED[x] = Accu.INITS[x]()

    @staticmethod
    def get_num(x=""):
        return Accu.NAMED[x]

    @staticmethod
    def add_num(x, f, init, func, inplace=False):
        if x not in Accu.NAMED:
            Accu.NAMED[x] = init()
            Accu.INITS[x] = init
            Accu.METHODS[x] = func
        if inplace:
            Accu.METHODS[x](Accu.NAMED[x], f)
        else:
            Accu.NAMED[x] = Accu.METHODS[x](Accu.NAMED[x], f)

# helpers #
# printing functions
def printing(s, func="info"):
    print(s)

def fatal(s):
    printing(s)
    printing("! FATAL, exit.")
    sys.exit()

def print_date():
    printing("--- Current time is %s." % (time.ctime()))

# ---------------------
def score_random(base):
    if sys.argv[-1] == "tr":    # total random
        x = random.random()
    elif sys.argv[-1] == "pr":  # add random
        x = base + random.random()
    elif sys.argv[-1] == "zz":  # zero out
        x = base*0.
    else:
        x = base
    return x

def debug_quiet():
    return not sys.argv[-1]=="debug"

def batch_run(pss):
    # manual batching
    hiddens = [[dy.inputVector([random.random() for _ in range(N_HIDDEN)])] for _ in range(N_BATCH)]
    scores = [[dy.scalarInput(0.)] for _ in range(N_BATCH)]
    nexts = [[0] for _ in range(N_BATCH)]   # start symbol
    for _ in range(N_STEP):
        # prepare
        batch_hiddens = []
        batch_inputs = []
        for i in range(N_BATCH):
            batch_hiddens += hiddens[i]
            batch_inputs += nexts[i]
        batch_hiddens_cat = dy.concatenate_to_batch(batch_hiddens)
        batch_embeds = dy.lookup_batch(lp, batch_inputs)
        calc_hiddens = gru(batch_embeds, batch_hiddens_cat)
        calc_outputs = pss * calc_hiddens
        # fire --- forward
        with Timer(name="forward", cname="f", quiet=debug_quiet()):
            val_outputs = calc_outputs.value()
        # next
        batch_base_index = 0
        val_outputs = np.reshape(val_outputs, (-1, N_WORDS))
        for i in range(N_BATCH):
            lit_scores = [x.value() for x in scores[i]]
            exp_scores = []
            for n, f in enumerate(lit_scores):
                exp_scores += [f+s for s in val_outputs[batch_base_index+n]]
            exp_scores = [score_random(b) for b in exp_scores]
            Accu.add_num("sort", len(exp_scores), lambda:0., lambda x,y:x+y)
            # Accu.add_num("sort_insts", exp_scores, lambda:[], lambda x,y:x.append(y), True)
            with Timer(name="sort", cname="sort", quiet=True):
                sort_args = np.argsort(exp_scores)[:N_BEAM]
            tmp_next_scores = []
            hiddens[i] = []
            nexts[i] = []
            for n, ind in enumerate(sort_args):
                hiddens[i].append(dy.pick_batch_elem(calc_hiddens, batch_base_index+ind//N_WORDS))
                nexts[i].append(ind%N_WORDS)
                this_calc_outputs = dy.pick_batch_elem(calc_outputs, batch_base_index+ind//N_WORDS)
                tmp_next_scores.append(scores[i][ind//N_WORDS]+dy.pick(this_calc_outputs, ind%N_WORDS))
            scores[i] = tmp_next_scores
            batch_base_index += len(lit_scores)
    # backward
    loss = 0.
    for i in range(N_BATCH):
        nn = len(scores[i])
        loss += scores[i][0] - scores[i][nn-1]
    with Timer(name="backward", cname="b", quiet=debug_quiet()):
        loss.backward()

def nonbatch_run(pss):
    hiddens = [[dy.inputVector([random.random() for _ in range(N_HIDDEN)])] for _ in range(N_BATCH)]
    scores = [[dy.scalarInput(0.)] for _ in range(N_BATCH)]
    nexts = [[0] for _ in range(N_BATCH)]   # start symbol
    for _ in range(N_STEP):
        batch_hiddens = [[] for _ in range(N_BATCH)]
        batch_outputs = [[] for _ in range(N_BATCH)]
        # prepare
        for i in range(N_BATCH):
            hs, ss, ns = hiddens[i], scores[i], nexts[i]
            for hh, nn in zip(hs, ns):
                batch_hiddens[i].append(gru(dy.lookup(lp, nn), hh))
                batch_outputs[i].append(pss * batch_hiddens[i][-1])
        # fire --- forward
        with Timer(name="forward", cname="f", quiet=debug_quiet()):
            batch_outputs[-1][-1].value()
        # next
        for i in range(N_BATCH):
            lit_scores = [x.value() for x in scores[i]]
            exp_scores = []
            for n, f in enumerate(lit_scores):
                exp_scores += [f+s for s in batch_outputs[i][n].value()]
            exp_scores = [score_random(b) for b in exp_scores]
            Accu.add_num("sort", len(exp_scores), lambda:0., lambda x,y:x+y)
            # Accu.add_num("sort_insts", exp_scores, lambda:[], lambda x,y:x.append(y), True)
            with Timer(name="sort", cname="sort", quiet=True):
                sort_args = np.argsort(exp_scores)[:N_BEAM]
            tmp_next_scores = []
            hiddens[i] = []
            nexts[i] = []
            for n, ind in enumerate(sort_args):
                hiddens[i].append(batch_hiddens[i][ind//N_WORDS])
                nexts[i].append(ind%N_WORDS)
                tmp_next_scores.append(scores[i][ind//N_WORDS]+dy.pick(batch_outputs[i][ind//N_WORDS], ind%N_WORDS))
            scores[i] = tmp_next_scores
    # backward
    loss = 0.
    for i in range(N_BATCH):
        nn = len(scores[i])
        loss += scores[i][0] - scores[i][nn-1]
    with Timer(name="backward", cname="b", quiet=debug_quiet()):
        loss.backward()

def test_sort(ll=None):
    lists = ll if ll is not None else [[score_random(random.random()) for _ in range(N_WORDS*N_BEAM)] for _ in range(N_BATCH*N_STEP)]
    for x in lists:
        with Timer(cname="sort", quiet=True):
            ktop = np.argsort(x)[:N_BEAM]
    Timer.show_accu()
    Timer.del_accu()

# !! mysterious bug of the sorting time ...
# -- maybe just inaccuracy of time recording
print(sys.argv)
batched = (sys.argv[1] == "y")
print(batched)
#test_sort()
for i in range(N_ITER):
    random.seed(12345)
    np.random.seed(12345)
    with Timer(name="Iter %s"%i, print_date=True) as _:
        with Timer(cname="all", quiet=True):
            for _ in range(N_BATCHES_ITER):
                dy.renew_cg()
                gru.ingraph()
                pss = dy.parameter(ss)
                if batched:
                    batch_run(pss)
                else:
                    nonbatch_run(pss)
    Timer.show_accu()
    Timer.del_accu()
    # with open("sorts-%s.iter%s.json" % (sys.argv[1], i), 'w') as f:
    #     f.write(json.dumps(Accu.get_num("sort_insts")))
    # test_sort(Accu.get_num("sort_insts"))
    # Accu.del_nums("sort_insts")
print("done.")
