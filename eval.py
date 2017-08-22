import utils
import os, subprocess

def evaluate(output, gold, metric):
    eva = {"bleu":_eval_bleu}[metric]
    return eva(output, gold)

def _eval_bleu(output, gold):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.join(dir_name, "tools", "multi-bleu.perl")
    p = subprocess.Popen("%s -lc %s < %s"%(script_name,gold,output), shell=True, stdout=subprocess.PIPE)
    line = p.stdout.readlines()[-1]
    utils.printing(line, func="info")
    b = float(line.split()[2][:-1])
    return b
