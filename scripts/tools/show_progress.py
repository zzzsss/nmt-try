# analyse the progress logging

import sys, json

def main():
    def _getv(z):   # bleu scores are >=0
        if z is None:
            return -1
        else:
            return z[0]
    def _gets(z):
        if type(z) != str:
            return z[0]
        else:
            return z
    with open(sys.argv[1], 'r') as fd:
        d = json.load(fd)
    best_point, best_score = "NOPE", None
    bad_points = d["bad_points"] + ["NOPE"]
    anneal_restarts_points = d["anneal_restarts_points"] + ["NOPE"]
    for ss, score in zip(d["hist_points"], d["hist_scores"]):
        print("%s(%s) || best: %s(%s)" % (ss, score, best_point, best_score))
        if _getv(score) > _getv(best_score):
            best_point, best_score = ss, score
        if ss == _gets(bad_points[0]):
            bad_points = bad_points[1:]
            print("== bad point")
        if ss == _gets(anneal_restarts_points[0]):
            anneal_restarts_points = anneal_restarts_points[1:]
            print("-- restart point")

if __name__ == '__main__':
    main()
