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


# -------------------------

def main():
    task_name = sys.argv[1]
    task = globals()[task_name]
    run(task)
    zhold(task["GPU"])

if __name__ == "__main__":
    main()
