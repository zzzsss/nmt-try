import os
import subprocess

from zl import utils
from . import mt_args

def evaluate(output, gold, metric, process_gold=False):
    eva = {"bleu":_eval_bleu,"ibleu":lambda _1,_2,_3:_eval_bleu(_1,_2,_3,True), "nist":_eval_nist}[metric]
    return eva(output, gold, process_gold)

def _get_lang(gold_fn):
    # current:
    cands = ["en", "fr", "de", "zh"]
    for c in cands:
        if c in gold_fn:
            return c
    utils.zlog("Unknown target languages for evaluating!!", func="warn")
    return "en"

def _eval_bleu(output, gold, process_gold, lowercase=False):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    restore_name = os.path.join(dir_name, "..", "scripts", "restore.sh")
    script_name = os.path.join(dir_name, "..", "scripts", "moses", "multi-bleu.perl")
    # zmt_name = os.path.join(dir_name, "..")  # todo(warn) need to find mosesdecoder for restore: default $ZMT is znmt/../
    # maybe preprocess
    # todo: special treatment for files with multiple references
    if str.isnumeric(gold[-1]):
        utils.zlog("Evaluating instead on %s to deal with multiple references of original %s." % (gold[:-1], gold), func="warn")
        gold = gold[:-1]
    elif process_gold:
        gold_res = "temp.somekindofhelpless.gold.restore"
        os.system("bash %s < %s > %s" % (restore_name, gold, gold_res))
        gold = gold_res
    maybe_lc = "-lc" if lowercase else ""
    p = subprocess.Popen("bash %s < %s | perl %s %s %s" % (restore_name, output, maybe_lc, script_name, gold), shell=True, stdout=subprocess.PIPE)
    line = p.stdout.readlines()
    utils.zlog("Evaluating %s to %s." % (output, gold), func="info")
    utils.zlog(str(line), func="score")
    b = float(line[-1].split()[2][:-1])
    return b

def _eval_nist(output, gold, process_gold):
    # => directly use eval_nist.sh
    raise NotImplementedError("need ref and other processing, calling outside.")

# well, eval directly using scripts
# def main():
#     opts = mt_args.init("eval")
#     evaluate(opts["files"][0], opts["files"][1], opts["eval_metric"])
#
# if __name__ == '__main__':
#     main()
