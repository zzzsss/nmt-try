#! /bin/python3

# decoding scripts

import os, subprocess, sys

printing = lambda x: print(x, flush=True)

def system(cmd, pp=True, ass=False, popen=True):
    if pp:
        printing("SYS-RUN: %s" % cmd)
    if popen:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        n = p.wait()
        output = p.stdout.read()
    else:
        n = os.system(cmd)
        output = None
    if pp:
        printing("SYS-OUT: %s" % output)
    if ass:
        assert n==0
    return output

def zhold(gpuid):
    CMD = "PYTHONPATH=$DY_ZROOT/gbuild/python python3 ../../znmt/run/zhold.py 2 --dynet-mem 4 --dynet-devices GPU:%d" % (gpuid,)
    system(CMD)

# ----------------------
def run(task):
    confs = task["CONFS"]
    for dataname in task["DATA_NAMES"]:
        for beam_size in task["BEAM_SIZES"]:
            for name, extras in confs:
                runname = name + "-%d" % beam_size
                extras = extras + " --beam_size %d" % beam_size
                if beam_size <= 10:
                    extras += " --test_batch_size 8"
                elif beam_size <= 20:
                    extras += " --test_batch_size 4"
                d = {"dataname":dataname, "runname":runname, "extras":extras, "dirname":task["DIR_NAME"]}
                CMD_RUN = task["CMD_RUN"] % d
                CMD_EVAL = task["CMD_EVAL"] % d
                #
                out0 = system(CMD_RUN)
                out1 = system(CMD_EVAL)
                printing("zzzzz %s %s" % (runname, out1))
    return None

# --------------

zd0213_ana = {
    "GPU":2,
    "CMD_RUN": "PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ../../znmt/test.py -v --test_batch_size 1 --eval_metric ibleu -o z.%(dataname)s.%(runname)s -t ../../zh_en_data/Test-set/%(dataname)s.src ../../zh_en_data/Reference-for-evaluation/%(dataname)s/%(dataname)s.ref0 -d %(dirname)s/src.v %(dirname)s/trg.v -m %(dirname)s/zbest.model --dynet-devices GPU:2 %(extras)s",
    "CMD_EVAL": "perl ../../znmt/scripts/multi-bleu.perl ../../zh_en_data/Reference-for-evaluation/%(dataname)s/%(dataname)s.ref < z.%(dataname)s.%(runname)s",
    "DATA_NAMES": ["nist_36", "nist_2002"],
    "DIR_NAME": "../../baselines/ze_ev_drop/",
    "BEAM_SIZES": [10,],
    "CONFS": [
        ("ana", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 1.0 --decode_latnbest --decode_latnbest_lreward 1.0 --decode_dump_sg"),
    ]
}

# -------------------------

def main():
    task_name = sys.argv[1]
    task = globals()[task_name]
    run(task)
    zhold(task["GPU"])

if __name__ == "__main__":
    main()
