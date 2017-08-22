# import numpy as np
# import sys, time, random, subprocess, os
# import dynet as dy
#
# random.seed(12345)
# np.random.seed(12345)
#
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

# # parameters
# N = 1000
# ITER = 10
# STEP = 50
# BATCH = 4000
# m = dy.Model()
# w = m.add_parameters((N,N))
# print(sys.argv)
# for i in range(ITER):
#     dy.renew_cg()
#     ww = dy.parameter(w)
#     h_rec = dy.inputVector([random.random() for _ in range(N*BATCH)])
#     h_add = dy.inputVector([random.random() for _ in range(N*BATCH)])
#     h_rec = dy.reshape(h_rec, (N,), batch_size=BATCH)
#     h_add = dy.reshape(h_add, (N,), batch_size=BATCH)
#     for s in range(STEP):
#         if s % 10 == 0:
#             report_statm("%s-%s"%(i,s))
#         h_rec = ww * h_rec + h_add
#         h_rec.value()
#     loss = dy.dot_product(h_rec, h_add)
#     loss = dy.sum_batches(loss)
#     report_statm("%s-%s"%(i,"before-backward"))
#     loss.backward()
#     report_statm("%s-%s"%(i,"after-backward"))

# ------------------
import dynet as dy
import random, os, gc
N = 1000
ITER = 10
STEP = 50
BATCH = 4000
m = dy.Model()
w = m.add_parameters((N,N))
ini = [random.random() for _ in range(N*BATCH)]
for i in range(ITER):
    dy.renew_cg()
    ww = dy.parameter(w)
    h_rec = dy.inputVector(ini)
    h_add = dy.inputVector(ini)
    h_rec = dy.reshape(h_rec, (N,), batch_size=BATCH)
    h_add = dy.reshape(h_add, (N,), batch_size=BATCH)
    for s in range(STEP):
        h_rec = ww * h_rec + h_add
    loss = dy.dot_product(h_rec, h_add)
    loss = dy.sum_batches(loss)
    loss.backward()
    print("Step: %s-end" % (i,))
    os.system("cat /proc/%s/status | grep VmRSS" % os.getpid())
    os.system("nvidia-smi | grep %s" % os.getpid())
    gc.collect()

# import dynet as dy
# import random, os
# ITER = 50
# N = 10000
# BS = 10000
# dy.renew_cg()
# for i in range(ITER):
#     dy.renew_cg()
#     h = dy.inputVector([random.random() for _ in range(N)])
#     z = dy.pick_batch_elems(h, [0 for _ in range(BS)])
#     ll = dy.sum_batches(dy.sum_elems(z))
#     zz = ll.value()
#     print("Step: %s-end" % (i,))
#     os.system("cat /proc/%s/status | grep VmRSS" % os.getpid())
#     os.system("nvidia-smi | grep %s" % os.getpid())
