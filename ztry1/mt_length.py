# about the length of output sentence
import numpy as np
from zl import layers, utils
from sklearn import linear_model

# how to take lengths into account (simple scale)
class MTLengther(object):
    @staticmethod
    def get_scaler_f(method, alpha):
        if alpha <= 0.:
            return (lambda l: 1.)
        else:
            return {"norm": (lambda l: 1. / pow(l, alpha)),
                    "google": (lambda l: 1. * pow(6, alpha) / pow(5+l, alpha))}[method]

# testing time or training a length fitter
class MTNormer(object):
    pass

class LinearGaussain(layers.Layer):
    SCALE_LENGTH = 100.0
    INIT_WIDTH = 3

    @staticmethod
    def trans_len(l):
        return l / LinearGaussain.SCALE_LENGTH

    @staticmethod
    def fit_once(data_iter):
        # first fitting a simple one: y = gaussian(ax+b, sigma), here not including xenc for that will be too large
        with utils.Timer(tag="Fit-length-once", print_date=True):
            # 1. collect length
            x, y = [], []
            for insts in data_iter.arrange_batches():
                for one in insts:
                    x.append(LinearGaussain.trans_len(len(one[0])))
                    y.append(LinearGaussain.trans_len(len(one[1])))
            ll = len(x)
            x1, y1 = np.array(x, dtype=np.float32).reshape((-1,1)), np.array(y, dtype=np.float32)
            # 2. fit linear model
            regr = linear_model.LinearRegression()
            regr.fit(x1, y1)
            a, b = regr.coef_, regr.intercept_
            # 3. fit sigma
            x1.reshape((-1,))
            errors = a*x1+b - y1
            sigma, mu = None, None
            ret = (a, b, sigma, mu)
            del x, y, x1, y1
            utils.zlog("Fitting Length with %s sentences and get %s." % (ll, ret), func="score")
        return ret

    def __init__(self, model, xlen, xadd, xback):
        super(LinearGaussain, self).__init__(model)
        self.xlen = xlen
        self.xadd = xadd
        self.xback = xback
        # params
        self.W = self._add_params((1, xlen))
        self.V = self._add_params((1, 1))
        self.B = self._add_params((1,))
        self.sigma = self._add_params((1,), init=np.array([LinearGaussain.INIT_WIDTH/LinearGaussain.SCALE_LENGTH,], dtype=np.float32))

