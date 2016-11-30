# -*- coding: utf-8 -*-

import sys
import scipy.stats

from util         import *
from pylab        import *
from numpy        import *
from numpy.random import *
from node         import *


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))
# シグモイド関数


def invsigmoid(z):
    return log(z) - log(1.0 - z)
# シグモイド関数の逆関数(ロジット関数)


def sigmoidln(x):
    return -log(1.0 + exp(-x))
# 対数シグモイド関数    


def normpdfln(x, m, std):
    return sum(-0.5 * log(2 * pi) - log(std) - 0.5 * ((x - m) / std)**2)
# 対数正規分布

class Logistic(Node):
    # tssbのサブクラス
    init_mean    = 0.0
    min_drift    = 0.01
    max_drift    = 1.0
    hmc_accepts  = 1
    hmc_rejects  = 1

    def __init__(self, parent=None, dims=1, tssb=None, drift=1.0):
        super(Logistic, self).__init__(parent=parent, tssb=tssb)

        if parent is None:
            self.dims   = dims # 256
            self._drift = drift * ones((dims)) # init_drift    = 0.1　初期値
            self.params = zeros((self.dims)) # array[0, 0, 0, ..]

        else:
            self.dims   = parent.dims
            self.params = self.drift() * randn(self.dims) + parent.params
        self._cache_sigmoidln() # 256個の値 

    def _cache_sigmoidln(self):
        self._sigln    = sigmoidln(self.params[newaxis, :]) # 軸追加するやつ
        self._negsigln = sigmoidln(-self.params[newaxis, :])

    def drift(self):
        if self.parent() is None:
            return self._drift
        else:
            return self.parent().drift()

    def sample(self, args):
        num_data = args['num_data'] if args.has_key('num_data') else 1
        return rand(num_data, self.dims) < sigmoid(self.params[newaxis, :])

    def resample_params(self):
        data       = self.get_data()
        counts     = sum(data, axis=0)
        num_data   = data.shape[0]
        drifts     = self.drift()

        def logpost(params):
            # Generalised Gaussian Diffusions
            if self.parent() is None:
                llh = normpdfln(params, self.init_mean, drifts)
            else:
                llh = normpdfln(params, self.parent().params, drifts)
            llh = llh + sum(counts * sigmoidln(params)) + sum((num_data - counts) * sigmoidln(-params))
            for child in self.children():
                llh = llh + normpdfln( child.params, params, drifts)
            return llh

        def logpost_grad(params):
            if self.parent() is None:
                grad = -(params - self.init_mean) / drifts**2
            else:
                grad = -(params - self.parent().params) / drifts**2

            probs = sigmoid(params)
            grad = grad + counts * (1.0 - probs) - (num_data - counts) * probs

            for child in self.children():
                grad = grad + (child.params - params) / drifts**2
            return grad

        # slice sampler
        if rand() < 0.1:
            self.params = slice_sample(self.params, logpost, step_out=True, compwise=True)

        else:
            self.params, accepted = hmc(self.params, logpost, logpost_grad, 25, exponential(0.05))
            Logistic.hmc_rejects += 1 - accepted
            Logistic.hmc_accepts += accepted

        self._cache_sigmoidln()

    def resample_hypers(self):
        if self.parent() is not None:
            raise Exception("Can only update hypers from root!")

        def logpost(drifts):
            if any(drifts < self.min_drift) or any(drifts > self.max_drift):
                return -inf
            def loglh(root):
                llh = 0.0
                for child in root.children():
                    llh = llh + normpdfln(child.params, root.params, drifts)
                    llh = llh + loglh(child)
                return llh
            return loglh(self) + normpdfln(self.params, self.init_mean, drifts)

        self._drift = slice_sample(self._drift, logpost, step_out=True, compwise=True)

    def logprob(self, x):
        return sum(x*self._sigln + (1.0-x)*self._negsigln)

    def complete_logprob(self):
        return self.logprob(self.get_data())