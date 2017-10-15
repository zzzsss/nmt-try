# some tmp testings

# test1: init from numpy
import dynet as dy
import numpy as np
np.random.seed(12345)
m = dy.Model()
shape = (2, 3)
x = np.random.rand(*shape)
z1 = x.copy().ravel(order="C")
z2 = x.copy().ravel(order="F")
p = m.add_parameters(shape, init=dy.NumpyInitializer(x))
