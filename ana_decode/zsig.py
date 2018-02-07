# Paired Bootstrap Resampling

# python *.py f0 f1 golds

import sys, math
from nltk.translate.bleu_score import closest_ref_length, brevity_penalty, modified_precision
import numpy as np

def read(fname):
    with open(fname) as f:
        lines = [s.split() for s in f]
    return lines

def get_rec(refs, hyp):
    rec = [(closest_ref_length(refs, len(hyp)), len(hyp))]
    for i in [1,2,3,4]:
        f = modified_precision(refs, hyp, i)
        rec.append((f.numerator, f.denominator))
    return rec

def get_bleu(recs):
    len_ref = sum(r[0][0] for r in recs)
    len_hyp = sum(r[0][1] for r in recs)
    bp = brevity_penalty(len_ref, len_hyp)
    prs = []
    for i in [1,2,3,4]:
        prs.append(sum(r[i][0] for r in recs)/sum(r[i][1] for r in recs))
    s = (0.25*math.log(p_i) for p_i in prs)
    s = bp * math.exp(math.fsum(s))
    return s

def main():
    np.random.seed(12345)
    f0, f1 = sys.argv[1], sys.argv[2]
    refs = sys.argv[3:]
    s0, s1 = read(f0), read(f1)
    srefs = [read(r) for r in refs]
    # check
    size = len(s0)
    assert size == len(s1)
    for sr in srefs:
        assert size == len(sr)
    # calculate
    rec0, rec1 = [], []
    for i in range(size):
        cur_refs = [r[i] for r in srefs]
        rec0.append(get_rec(cur_refs, s0[i]))
        rec1.append(get_rec(cur_refs, s1[i]))
    # compare
    print("BLEU %f vs %f" % (get_bleu(rec0), get_bleu(rec1)))
    TIMES = 1000
    cmps = [0,0,0]
    for _ in range(TIMES):
        idxs = np.random.choice(size, size)
        r0 = [rec0[i] for i in idxs]
        r1 = [rec1[i] for i in idxs]
        b0 = get_bleu(r0)
        b1 = get_bleu(r1)
        if b0>b1:
            cmps[0]+=1
        elif b0<b1:
            cmps[-1]+=1
        else:
            cmps[1]+=1
    print("Results: %s" % cmps)

if __name__ == "__main__":
    main()
