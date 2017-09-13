import utils, args
import os, subprocess

def evaluate(output, gold, metric, opts=None):
    eva = {"bleu":_eval_bleu}[metric]
    return eva(output, gold)

def _eval_bleu(output, gold):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    restore_name = os.path.join(dir_name, "scripts", "restore.sh")
    script_name = os.path.join(dir_name, "scripts", "multi-bleu.perl")
    # have to specify the $ZMT env
    p = subprocess.Popen("ZMT=.. bash %s < %s | perl %s -lc %s" % (restore_name, output, script_name, gold), shell=True, stdout=subprocess.PIPE)
    line = p.stdout.readlines()
    utils.printing(str(line), func="info")
    b = float(line[-1].split()[2][:-1])
    return b

if __name__ == '__main__':
    opts = args.init("eval")
    utils.init_print()
    evaluate(opts["files"][0], opts["files"][1], opts["eval_metric"])
