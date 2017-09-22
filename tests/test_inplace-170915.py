# import dynet as dy
# import numpy as np
# import random
# N = 100
# Ns = 10
# m = dy.Model()
# ini = [random.random() for _ in range(N)]
# w1 = m.add_parameters((N,), dy.NumpyInitializer(np.array(ini)))
# w2 = m.add_parameters((N,), dy.NumpyInitializer(np.array(ini)))
# tr = dy.SimpleSGDTrainer(m)
# assert all(w1.as_array() == w2.as_array())
# for inplaced in [True, False]:
#     wp = dy.parameter(w1 if inplaced else w2)
#     x1 = dy.inputVector(ini)
#     x2 = dy.reshape(x1, (Ns, Ns), inplaced)
#     wp2 = dy.reshape(wp, (Ns, Ns), inplaced)
#     y1 = wp2 * x2
#     y2 = dy.reshape(y1, (N,), inplaced)
#     z1 = y2 + wp
#     z2 = dy.tanh(z1, inplaced)
#     loss = dy.sum_elems(z2)
#     loss.backward()
#     tr.update()
# assert all(w1.as_array() == w2.as_array())

# import dynet as dy
# import random
# import sys
# if sys.argv[1].startswith("in"):
#     inplaced = True
#     print("inplace", file=sys.stderr)
# else:
#     inplaced = False
#     print("notinplace", file=sys.stderr)
# Ni = 5
# Nh = 5
# BS = 1
# m = dy.Model()
# random.seed(12345)
# ini1 = [([random.random() for _1 in range(Ni)], 1) for _ in range(BS)]
# ini0 = [([random.random() for _1 in range(Ni)], 0) for _ in range(BS)]
# ws = {
# "w1": m.add_parameters((Nh, Ni)), "b1": m.add_parameters((Nh,)),
# "w2":m.add_parameters((1, Nh)), "b2":m.add_parameters((1,))
# }
# ps = {}
# for k in ws:
#     ps[k] = dy.parameter(ws[k])
# tr = dy.SimpleSGDTrainer(m)
# losses = []
# exprs = {"h1":[], "h1t":[], "h1d":[], "out":[]}
# for x, y in ini0+ini1:
#     h0 = dy.inputVector(x)
#     h1 = dy.affine_transform([ps["b1"], ps["w1"], h0])
#     h1t = dy.tanh(h1)
#     h1d = dy.dropout(h1t, 0.2, inplaced)
#     out = dy.affine_transform([ps["b2"], ps["w2"], h1d])
#     losses.append(dy.square(y-out))
#     tmp = locals().copy()
#     for k, v in tmp.items():
#         if k in exprs:
#             exprs[k].append(v)
# loss = dy.esum(losses)
# loss.backward()
# for k in exprs:
#     for v in exprs[k]:
#         try:
#             print("%s: %s" % (k, v.gradient()))
#         except:
#             print("Cannot get gradietn of %s" % k)
# tr.update()
# print([ws[w].as_array() for w in ws])

import dynet as dy
import random, sys
if sys.argv[1].startswith("in"):
    inplaced = True
    print("inplace", file=sys.stderr)
else:
    inplaced = False
    print("notinplace", file=sys.stderr)
Ni, Nh, BS = 5, 5, 1
m = dy.Model()
random.seed(12345)
ini1 = [([random.random() for _1 in range(Ni)], 1) for _ in range(BS)]
ini0 = [([random.random() for _1 in range(Ni)], 0) for _ in range(BS)]
ws = { "w1": m.add_parameters((Nh, Ni)), "b1": m.add_parameters((Nh,)), "w2":m.add_parameters((1, Nh)), "b2":m.add_parameters((1,))}
ps = {}
for k in ws:
    ps[k] = dy.parameter(ws[k])
tr = dy.SimpleSGDTrainer(m)
losses = []
for _ in range(100):
    for x, y in ini0+ini1:
        h0 = dy.inputVector(x)
        h1 = dy.affine_transform([ps["b1"], ps["w1"], h0])
        h1.set_rewritable(True)
        h1t = dy.tanh(h1, inplaced=inplaced)
        h1d = dy.dropout(h1t, 0.2)
        out = dy.affine_transform([ps["b2"], ps["w2"], h1d])
        losses.append(dy.square(y-out))
    loss = dy.esum(losses)
    loss.backward()
    tr.update()
for w in sorted(ws.keys()):
    print("%s: %s" % (w, ws[w].as_array()))
