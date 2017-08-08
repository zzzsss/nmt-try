import numpy as np
import sys, time, random, subprocess, os
import dynet as dy

random.seed(12345)
np.random.seed(12345)

def printing(s, func="info"):
    print(s)

def get_statm():
    with open("/proc/self/statm") as f:
        rss = (f.read().split())        # strange!! readline-nope, read-ok
        mem0 = str(int(rss[1])*4/1024) + "MiB"
    try:
        p = subprocess.Popen("nvidia-smi | grep -E '%s.*MiB'" % os.getpid(), shell=True, stdout=subprocess.PIPE)
        line = p.stdout.readlines()
        mem1 = line[-1].split()[-2]
    except:
        mem1 = "0MiB"
    return mem0, mem1

def report_statm(s):
    printing(str(get_statm())+"at step %s"%s)

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

# parameters
N = 2000
ITER = 10
STEP = 50
BATCH = 80

m = dy.Model()
g = GruNode(m, N, N)

def zzzz(*s):
    pass

print(sys.argv)
for i in range(ITER):
    random.seed(12345)
    np.random.seed(12345)
    # renew
    dy.renew_cg()
    g.ingraph()
    h_rec = dy.inputVector([random.random() for z in range(N*BATCH)])
    h_add = dy.inputVector([random.random() for z in range(N*BATCH)])
    h_rec = dy.reshape(h_rec, (N,), batch_size=BATCH)
    h_add = dy.reshape(h_add, (N,), batch_size=BATCH)
    for s in range(STEP):
        report_statm("%s-%s"%(i,s))
        h_rec = g(h_add, h_rec)
        zzzz(h_rec, h_add)
        zzzz(h_rec.value())
    loss = dy.dot_product(h_rec, h_add)
    loss = dy.sum_batches(loss)
    loss.backward()
