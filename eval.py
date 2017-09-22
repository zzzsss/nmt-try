import utils, args
import os, subprocess

def evaluate(output, gold, metric, process_gold=False):
    eva = {"bleu":_eval_bleu}[metric]
    return eva(output, gold, process_gold)

def _eval_bleu(output, gold, process_gold):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    restore_name = os.path.join(dir_name, "scripts", "restore.sh")
    script_name = os.path.join(dir_name, "scripts", "multi-bleu.perl")
    zmt_name = os.path.join(dir_name, "..")
    # maybe preprocess
    if process_gold:
        gold_res = "temp.somekindofhelpless.gold.restore"
        os.system("ZMT=%s bash %s < %s > %s" % (zmt_name, restore_name, gold, gold_res))
        gold = gold_res
    p = subprocess.Popen("ZMT=%s bash %s < %s | perl %s %s" % (zmt_name, restore_name, output, script_name, gold), shell=True, stdout=subprocess.PIPE)
    line = p.stdout.readlines()
    utils.printing(str(line), func="score")
    b = float(line[-1].split()[2][:-1])
    return b

if __name__ == '__main__':
    opts = args.init("eval")
    utils.init_print()
    evaluate(opts["files"][0], opts["files"][1], opts["eval_metric"])
