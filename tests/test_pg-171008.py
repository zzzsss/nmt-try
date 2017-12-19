#!/bin/python3

import dynet as dy
import numpy as np
import sys

np.random.seed(1)

def get_data(n_x, n_y, n_point, n_sample):
    scale, stddev = 10, 0.1
    centers = [[np.random.rand(n_x)*scale for _j in range(n_point)] for _i in range(n_y)]
    # centers = [[[0.,0.], [1.,1.]], [[1.,0.], [0.,1.]]]
    # centers = [[[0.,0.], [0.,1.]], [[1.,1.], [1.,1.]]]
    retx = []
    rety = []
    for idx, cs in enumerate(centers):
        for i in [int(one*n_point) for one in np.random.rand(n_sample)]:
            retx += [v for v in cs[i]+np.random.randn(n_x)*stddev]
            rety.append(idx)
    return retx, rety, centers

def get_model(n_x, n_y, n_hidden):
    mm = dy.Model()
    params = {}
    params["w0"] = mm.add_parameters((n_hidden, n_x))
    # params["b0"] = mm.add_parameters((n_hidden,))
    params["w1"] = mm.add_parameters((n_y, n_hidden))
    params["b1"] = mm.add_parameters((n_y,))
    def _call(x):
        iparams = {}
        for k in params:
            iparams[k] = dy.parameter(params[k])
        h0 = iparams["w0"] * x
        h0 = getattr(dy, sys.argv[2])(h0)
        h2 = iparams["w1"] * h0
        h2 += iparams["b1"]
        h3 = dy.softmax(h2)
        return h2, h3
    return mm, _call, params

def acc(preds, golds):
    assert len(preds) == len(golds)
    a = [1 if x==y else 0 for x, y in zip(preds, golds)]
    print("ACC: %s/%s=%s" % (sum(a), len(a), sum(a)/len(a)))

def build_loss(scores, probs_val, y, way):
    bsize = len(y)
    if way < 0:
        mu = dy.pickneglogsoftmax_batch(scores, y)
    else:
        target = []
        for s, t in zip(probs_val, y):
            if way==0:
                target += [-1.+one if i==t else one for i,one in enumerate(s)]
            elif way==1:
                target += [-1.+one if i==t else 0. for i,one in enumerate(s)]
            elif way==2:
                target += [-1 if i==t else 1./(len(s)-1) for i,one in enumerate(s)] if np.argmax(s)!=t else [0. for i in range(len(s))]
            elif way==3:
                target += [-1 if i==t else 0. for i,one in enumerate(s)] if np.argmax(s)!=t else [0. for i in range(len(s))]
            elif way==4:    # no constrains
                target += [-1 if i==t else 0. for i,one in enumerate(s)]
        out = dy.inputVector(target)
        arr = dy.reshape(out, (len(probs_val[0]),), batch_size=bsize)
        mu = dy.cmult(arr, scores)
    loss = dy.sum_elems(mu)
    loss = dy.sum_batches(loss) / bsize
    return loss

def test():
    n_in = 2
    n_hidden = 20
    n_class = 2
    n_point = 5
    n_sample = 100
    n_iters = 1000
    way = int(sys.argv[1])
    x, y, centers = get_data(n_in, n_class, n_point, n_sample)
    print(centers)
    mm, ff, params = get_model(n_in, n_class, n_hidden)
    # tr = dy.SimpleSGDTrainer(mm, learning_rate=0.1)
    tr = dy.MomentumSGDTrainer(mm, learning_rate=0.1, mom=0.9)
    # start to train
    for i in range(n_iters):
        dy.renew_cg()
        ix = dy.inputVector(x)
        arr = dy.reshape(ix, (n_in,), batch_size=len(y))
        scores, probs = ff(arr)
        # acc
        probs_val = np.reshape(np.array(probs.value()), (len(y), n_class))
        preds = [np.argmax(one) for one in probs_val]
        if i%50==0:
            acc(preds, y)
        # gradients
        loss = build_loss(scores, probs_val, y, way)
        loss.backward()
        # update
        tr.update()
    # print
    for k in params:
        print("%s:%s" % (k, params[k].as_array()))

# findings:
# 1. non-linear, bring about twists in higher-dimension for identification
# 2. importance of balanced gradients: control scale?

test()
