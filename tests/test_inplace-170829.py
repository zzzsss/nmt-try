import dynet as dy
import numpy as np
import random
N = 100
Ns = 10
m = dy.Model()
ini = [random.random() for _ in range(N)]
w1 = m.add_parameters((N,), dy.NumpyInitializer(np.array(ini)))
w2 = m.add_parameters((N,), dy.NumpyInitializer(np.array(ini)))
tr = dy.SimpleSGDTrainer(m)
assert all(w1.as_array() == w2.as_array())
for inplaced in [True, False]:
    wp = dy.parameter(w1 if inplaced else w2)
    x1 = dy.inputVector(ini)
    x2 = dy.reshape(x1, (Ns, Ns), inplaced)
    wp2 = dy.reshape(wp, (Ns, Ns), inplaced)
    y1 = wp2 * x2
    y2 = dy.reshape(y1, (N,), inplaced)
    z1 = y2 + wp
    z2 = dy.tanh(z1, inplaced)
    loss = dy.sum_elems(z2)
    loss.backward()
    tr.update()
assert all(w1.as_array() == w2.as_array())

if 1:
    # failing situation 1.1: cannot restore value for revert
    import dynet as dy
    x = dy.zeroes(100)
    dy.cg_checkpoint()
    y = dy.tanh(x, True)
    y.value()
    dy.cg_revert()

    # failing situation 1.2: cannot restore value for get_value
    import dynet as dy
    x = dy.zeroes(100)
    y = dy.tanh(x, True)
    y.value()
    print(x.value())

    # failing situation 1.3: cannot use the node whose value has been WRITE-borrowed
    import dynet as dy
    x = dy.zeroes(100)
    y = dy.tanh(x, True)
    z = x + 1

    # failing situation 2.1: *-inplace + WRITE-inplace
    import dynet as dy
    x = dy.zeroes(100)
    y = dy.tanh(x, True)
    y2 = dy.tanh(y, True)

    # failing situation 2.2: used node (by plain nodes or READ-inplace nodes) + WRITE-inplace
    import dynet as dy
    x = dy.zeroes(100)
    z = x + 1
    y = dy.tanh(x, True)

    # ok situation: multiple READ-inplace
    import dynet as dy
    x = dy.zeroes(100)
    x1 = dy.reshape(x, (10,10), True)
    x2 = dy.reshape(x, (10,10), True)
    x11 = dy.reshape(x1, (100,), True)
    for z in [x,x1,x2,x11]:
        print(z.dim())
