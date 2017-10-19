import os
import subprocess

from . import utils
from . import args


def evaluate(output, gold, metric, process_gold=False):
    eva = {"bleu":_eval_bleu, "nist":_eval_nist}[metric]
    return eva(output, gold, process_gold)

def _get_lang(gold_fn):
    # current:
    cands = ["en", "fr", "de"]
    for c in cands:
        if c in gold_fn:
            return c
    utils.printing("Unknown target language for evaluating!!", func="warn")
    return "en"

def _eval_bleu(output, gold, process_gold):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    restore_name = os.path.join(dir_name, "..", "scripts", "restore.sh")
    script_name = os.path.join(dir_name, "..", "scripts", "moses", "multi-bleu.perl")
    # zmt_name = os.path.join(dir_name, "..")  # todo(warn) need to find mosesdecoder for restore: default $ZMT is znmt/../
    # maybe preprocess
    if process_gold:
        gold_res = "temp.somekindofhelpless.gold.restore"
        os.system("bash %s < %s > %s" % (restore_name, gold, gold_res))
        gold = gold_res
    p = subprocess.Popen("bash %s < %s | perl %s %s" % (restore_name, output, script_name, gold), shell=True, stdout=subprocess.PIPE)
    line = p.stdout.readlines()
    utils.printing("Evaluating %s to %s." % (output, gold), func="info")
    utils.printing(str(line), func="score")
    b = float(line[-1].split()[2][:-1])
    return b

def _eval_nist(output, gold, process_gold):
    # => directly use eval_nist.sh
    raise NotImplementedError("need ref and other processing, calling outside.")

def main():
    opts = args.init("eval")
    utils.init_print()
    evaluate(opts["files"][0], opts["files"][1], opts["eval_metric"])

if __name__ == '__main__':
    main()
