import os
import sys
import time
import cPickle
import sigmoid_clf
import pandas as pd
from numpy        import *
from numpy.random import *
from logistic     import *
from util         import *
from my_tssb         import *

rand_seed     = 1234
dp_alpha      = 25.0
dp_gamma      = 1.0
init_drift    = 0.1
alpha_decay   = 0.25
codename      = os.popen('./random-word').read().rstrip()
print "Codename: ", codename
start_time = time.time()

seed(rand_seed)

print """
----------------------------
data load start
----------------------------
"""
# ------------------------------------------

filename = './input/autoencoder_train.vector'
fh = open(filename)
image_data = cPickle.load(fh)
fh.close()

image_data = sigmoid_clf.feature_trans(image_data)
dims = image_data.shape[1]
root = Logistic( dims=dims, drift=init_drift)

train_dir = "./input/autoencoder_train.target"
fh = open(train_dir)
target = cPickle.load(fh)
fh.close()
category_index = pd.read_csv("./input/category_index_df.csv")
category_index.columns = ['category_id', 'class1', 'class2', 'class3']

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

print """
----------------------------
tssb class load
----------------------------
"""
tssb = TSSB(dp_alpha=dp_alpha, dp_gamma=dp_gamma, alpha_decay=alpha_decay, root_node=root, data=image_data)

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

print """
----------------------------
make initial class
----------------------------
"""
tssb.make_category(max_category1=5, max_category2=120, max_category3=20)

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

print """
----------------------------
add predata
----------------------------
"""
tssb.add_pre_data(target, category_index)

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

print """
----------------------------
data shape is %s
----------------------------
""" % str(tssb.data.shape)

print """
----------------------------
assignments length is %s
----------------------------
""" % str(len(tssb.assignments))

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

print """
----------------------------
cull tree
----------------------------
"""
tssb.cull_tree()

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

# mcmc config
burnin = 10
num_samples = 40
checkpoint = 1000

dp_alpha_traces = zeros((num_samples, 1))
dp_gamma_traces = zeros((num_samples, 1))
alpha_decay_traces = zeros((num_samples, 1))
drift_traces = zeros((num_samples, dims))
cd_llh_traces = zeros((num_samples, 1))

intervals = zeros((7))
print """
----------------------------
Starting MCMC run...
----------------------------
"""

start_time = time.time()

for iter in range(-burnin, num_samples):
    elapsed_time = time.time() - start_time
    print "----------------------------"
    print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"
    print "----------------------------"

    times = [ time.time()]

    tssb.resample_node_hyper()
    times.append(time.time())

    tssb.resample_node_params()
    times.append(time.time())

    tssb.resample_hypers()
    times.append(time.time())

    tssb.resample_sticks()
    times.append(time.time())

    tssb.resample_stick_orders()
    times.append(time.time())

    if iter > 0:
        tssb.resample_hypers(dp_alpha=True, alpha_decay=True, dp_gamma=True)
    times.append(time.time())

    # intervals = intervals + diff(array(times))

    if iter >= 0:
        dp_alpha_traces[iter]    = tssb.dp_alpha
        dp_gamma_traces[iter]    = tssb.dp_gamma
        alpha_decay_traces[iter] = tssb.alpha_decay
        drift_traces[iter]       = root.drift()
        cd_llh_traces[iter]      = tssb.complete_data_log_likelihood()

    if iter > 0 and mod(iter, checkpoint) == 0:
        filename = "checkpoints/cifar100-50k-%s-%06d.pkl" % (codename, iter)
        fh = open(filename, 'w')
        cPickle.dump(tssb, fh)
        fh.close()
    if True or mod(iter, 10) == 0:
        (weights, nodes) = tssb.get_mixture()
        # check value
        print codename, "iter is %d" % iter
        print "nodes length is %s" % str(len(nodes))
        print "cd_llh_trace is %d" % cd_llh_traces[iter]
        print "drift mean = %d" % mean(root._drift)
        print "dp_alpha={0}: dp_gamma={1}: alpha_decay:{2}".format(tssb.dp_alpha, tssb.dp_gamma, tssb.alpha_decay) 
        # print "intervals", " ".join(map(lambda x: "%0.2f" % x, intervals.tolist()))
        print float(root.hmc_accepts) / (root.hmc_accepts + root.hmc_rejects)
        print "hmc accept = %d" % root.hmc_accepts
        print "hmc reject = %d" % root.hmc_rejects

        # intervals = zeros((7))
        if iter > 0 and argmax(cd_llh_traces[:iter + 1]) == iter:
            filename = "bests/tssb-%s-best.pkl" % (codename)
            fh = open(filename, 'w')
            cPickle.dump(tssb, fh)
            fh.close()

filename = "checkpoints/tssb-%s-final.pkl" % (codename)
fh = open(filename, 'w')
cPickle.dump({ 'tssb'               : tssb,
               'dp_alpha_traces'    : dp_alpha_traces,
               'dp_gamma_traces'    : dp_gamma_traces,
               'alpha_decay_traces' : alpha_decay_traces,
               'drift_traces'       : drift_traces,
               'cd_llh_traces'      : cd_llh_traces }, fh)
fh.close()


# test_dir = "input/autoencoder_test.vector"
# fh       = open(test_dir)
# test  = cPickle.load(fh)
# fh.close()

# test = sigmoid_clf.feature_trans(test)
# tssb.add_data(test[0:1])
# print "add_data"
# print "data shape is %s" % str(tssb.data.shape)
# print "assgnments length is %s" % str(len(tssb.assignments))
