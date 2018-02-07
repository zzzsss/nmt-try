# analyze the outputs

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import sys, numpy

# read from two files of non-verbose *s file: tokens with score

# def multi_to_one(files):
#     fds = [open(f) for f in files]
#     for line in zip(*fds):
#         for l in line:
#             print(l, end="")
#         print()
# multi_to_one(sys.argv[1:])

class Inst(object):
    def __init__(self, tokens, scores):
        assert len(tokens) == len(scores)
        assert len(tokens) > 0
        self.length = len(tokens)
        self.tokens = tokens
        self.scores = scores
        self.sbleus = [None] * self.length
        self.oracle_idx = None

    @property
    def best_toks(self):
        return self.tokens[0]

    @property
    def oracle_toks(self):
        return self.tokens[self.oracle_idx]

    @property
    def oracle_hit(self):
        return 1 if self.oracle_idx==0 else 0

    def __repr__(self):
        return str(self.tokens) + str(self.scores)

    def __str__(self):
        return self.__repr__()

# read multi file
def read_them(file):
    ret = []
    with open(file) as fd:
        tokens, scores = [], []
        for line in fd:
            fields = line.split()
            if len(fields) == 0:
                if len(tokens) > 0:
                    ret.append(Inst(tokens, scores))
                tokens, scores = [], []
                continue
            # read line
            one_toks = []
            for tok in fields[:-1]:
                tmp_splits = tok.split("|")
                if len(tmp_splits) > 1:
                    try:
                        ss = float(tmp_splits[-1])
                        tok = "|".join(tmp_splits[:-1])
                    except:
                        pass
                one_toks.append(tok)
            # maybe eos
            final_tok = fields[-1]
            final_score = None
            tmp_splits = final_tok.split("|")
            if len(tmp_splits) > 1:
                try:
                    ss = float(tmp_splits[-1])
                    final_score = ss
                except:
                    one_toks.append(final_tok)
            else:
                one_toks.append(final_tok)
            tokens.append(one_toks)
            scores.append(final_score)
        if len(tokens) > 0:
            ret.append(Inst(tokens, scores))
    return ret

def main():
    gold = read_them(sys.argv[1])
    num_insts = len(gold)
    r1 = read_them(sys.argv[2])
    r2 = read_them(sys.argv[3])
    assert num_insts == len(r1)
    assert num_insts == len(r2)
    # step0: calculate bleu scores
    print("step0: calculate bleu scores")
    smooth_f = SmoothingFunction().method2      # +1 smoothing
    for i, r in enumerate([r1,r2]):
        for inst, g in zip(r, gold):
            refs = g.tokens
            for idx, toks in enumerate(inst.tokens):
                sbs = sentence_bleu(refs, toks, smoothing_function=smooth_f)
                inst.sbleus[idx] = sbs
            inst.oracle_idx = int(numpy.argmax(inst.sbleus))
    # step1: overall comparisons
    print("step1: overall comparisons")
    all_refs = [x.tokens for x in gold]
    for i, r in enumerate([r1,r2]):
        with open("bs0_f%s.txt" % i, "w") as fd:
            for x in r:
                fd.write(" ".join(x.best_toks)+"\n")
        bs0 = corpus_bleu(all_refs, [x.best_toks for x in r])
        with open("bsb_f%s.txt" % i, "w") as fd:
            for x in r:
                fd.write(" ".join(x.oracle_toks)+"\n")
        bsb = corpus_bleu(all_refs, [x.oracle_toks for x in r])
        hits = sum(x.oracle_hit for x in r)
        print("-- For file %d, b0=%s, bb=%s, hits=%s/%s/%.3f" % (i, bs0, bsb, hits, num_insts, hits/num_insts))
    # step2: comparing each other
    print("step2: comparing each other")
    def _cmp_f(a, b):
        if a==b:
            return 0
        elif a>b:
            return 1
        else:
            return -1
    cmp_bleus = [0,0,0]
    cmp_scores_all = [0,0,0]
    cmp_scores_1better = [0,0,0]
    cmp_scores_2better = [0,0,0]
    same_best = 0
    for inst1, inst2 in zip(r1, r2):
        if inst1.tokens[0] == inst2.tokens[0]:
            same_best += 1
        bleu1, bleu2 = inst1.sbleus[0], inst2.sbleus[0]
        score1, score2 = inst1.scores[0], inst2.scores[0]
        c_b = _cmp_f(bleu1, bleu2)
        c_s = _cmp_f(score1, score2)
        cmp_bleus[c_b] += 1
        cmp_scores_all[c_s] += 1
        if c_b == 1:
            cmp_scores_1better[c_s] += 1
        elif c_b == -1:
            cmp_scores_2better[c_s] += 1
    for cc in [cmp_bleus, cmp_scores_all, cmp_scores_1better, cmp_scores_2better]:
        print("%s // %s" % (cc, [x/num_insts for x in cc]))
    print("%s // %s" % (same_best, same_best/num_insts))

if __name__ == '__main__':
    main()

#
# import sys
# from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
# hyp = [line.split() for line in open(sys.argv[1])]
# refs = [[line.split() for line in open(name)] for name in sys.argv[2:]]
# i, n = len(refs), len(refs[0])
# refs2 = [[refs[z][y] for z in range(i)] for y in range(n)]
# print(corpus_bleu(refs2, hyp))
