# -*- coding: utf-8 -*-
# node のハンドリングにおいて用いるスクリプト

import sys

from numpy.random import *


class Node(object):

    def __init__(self, parent=None, tssb=None):
        self.data      = set([])
        self._children = set([])
        self.tssb      = tssb
        # ここが入っていないのでは??

        if parent is not None:
            parent.add_child(self)
            self._parent = parent
        else:
            self._parent = None

    def kill(self):
        if self._parent is not None:
            self._parent._children.remove(self)

        self._parent   = None
        self._children = None

    def spawn(self):
        # 親の分布を継承する   
        return self.__class__(parent=self, tssb=self.tssb)

    def has_data(self):
        if len(self.data):
            return True
        else:
            for child in self._children:
                if child.has_data():
                    return True
        return False

    def num_data(self):
        return reduce(lambda x, y: x+y, map(lambda c: c.num_data(), self._children), len(self.data))

    def num_local_data(self):
        return len(self.data)

    def add_datum(self, id):
        self.data.add(id)

    def remove_datum(self, id):
        self.data.remove(id)

    def resample_params(self):
        pass

    def add_child(self, child):
        self._children.add(child)

    def remove_child(self, child):
        self._children.remove(child)

    def children(self):
        return self._children

    def get_data(self):
        return self.tssb.data[list(self.data), :]

    def logprob(self, x):
        return 0

    def data_log_likelihood(self):
        return self.complete_logprob()
    
    def sample(self, num_data=1):
        return rand(num_data, 2)

    def parent(self):        
        return self._parent

    def global_param(self, key):
        if self.parent() is None:
            return self.__dict__[key]
        else:
            return self.parent().global_param(key)

    def get_ancestors(self):
        if self._parent is None:
            return [self]
        # self = 自身のノード
        else:
            ancestors = self._parent.get_ancestors()
            ancestors.append(self)
            return ancestors
