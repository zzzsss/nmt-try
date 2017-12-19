import dynet as dy
import sys

if sys.argv[1] == "v":
    vec = True
else:
    vec = False
# print("Using vec?: %s" % vec)

N = 1000
BS = 1000
ITER = 100
m = dy.Model()
px = m.add_parameters((N, N))
input = dy.ones((N,), batch_size=BS)
pp = dy.parameter(px)

z = input
for i in range(ITER):
    z = pp * z
    z.forward()
    if not vec:
        k = z.npvalue()
        # print(k.shape)
    else:
        k = z.vec_value()
        # print(len(k))
