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
    CMD = "PYTHONPATH=$DY_ZROOT/gbuild/python python3 ../../znmt/run/zhold.py 3 --dynet-mem 4 --dynet-devices GPU:%d" % (gpuid,)
    system(CMD)

# ----------------------
def run(task):
    confs = task["CONFS"]
    for dataname in task["DATA_NAMES"]:
        for beam_size in [10, 12]:
            for name, extras in confs:
                runname = name + "-%d" % beam_size
                extras = extras + " --beam_size %d" % beam_size
                if beam_size <= 20:
                    extras += " --test_batch_size 4"
                d = {"dataname":dataname, "runname":runname, "extras":extras, "dirname":task["DIR_NAME"]}
                CMD_RUN = task["CMD_RUN"] % d
                CMD_EVAL = task["CMD_EVAL"] % d
                #
                out0 = system(CMD_RUN)
                out1 = system(CMD_EVAL)
                printing("zzzzz %s %s" % (runname, out1))
    return None

# -------------------------

TasksZhEn0207_all = {
    "GPU":0,
    "CMD_RUN": "PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ../../znmt/test.py -v --test_batch_size 1 --eval_metric ibleu -o z.%(dataname)s.%(runname)s -t ../../zh_en_data/Test-set/%(dataname)s.src ../../zh_en_data/Reference-for-evaluation/%(dataname)s/%(dataname)s.ref0 -d %(dirname)s/src.v %(dirname)s/trg.v -m %(dirname)s/zbest.model --dynet-devices GPU:0 %(extras)s",
    "CMD_EVAL": "perl ../../znmt/scripts/multi-bleu.perl ../../zh_en_data/Reference-for-evaluation/%(dataname)s/%(dataname)s.ref < z.%(dataname)s.%(runname)s",
    "DATA_NAMES": ["nist_2002", "nist_2003", "nist_2004", "nist_2006", "nist_2005", "nist_36", "nist_2008"],
    "DIR_NAME": "../../baselines/ze_ev_drop/",
    "CONFS": [
        # basic
        ("0a", "--pr_local_diff 2.3 --normalize_way none"),
        ("0b", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 1.0"),
        ("0c", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0"),
        # merge
        ("1a", "--pr_local_diff 2.3 --normalize_way none --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest"),
        ("1b", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest"),
        ("1c", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 1.0 --decode_latnbest --decode_latnbest_lreward 1.0"),
        ("1d", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 1.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_nalpha 1.0"),
        ("1e", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 1.0 --decode_latnbest --decode_latnbest_lreward 1.0"),
        ("1f", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 1.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_nalpha 1.0"),
    ]
}

TasksEnDE0207_all = {
    "GPU":2,
    "CMD_RUN": "PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ../../znmt/test.py -v --test_batch_size 1 -o z.%(dataname)s.%(runname)s.bpe -t ../../en_de_data_z5/%(dataname)s.bpe.en ../../en_de_data_z5/%(dataname)s.tok.de -d %(dirname)s/src.v %(dirname)s/trg.v -m %(dirname)s/zbest.model --dynet-devices GPU:2 %(extras)s",
    "CMD_EVAL": "ZMT=../.. bash ../../znmt/scripts/restore.sh <z.%(dataname)s.%(runname)s.bpe | tee z.%(dataname)s.%(runname)s | perl ../..//znmt/scripts/multi-bleu.perl ../../en_de_data_z5/%(dataname)s.tok.de",
    "DATA_NAMES": ["data2014", "data2015", "data2016", "data2012", "data2013"],
    "DIR_NAME": "../../baselines/ed_ev/",
    "CONFS": [
        # basic
        ("0a", "--pr_local_diff 2.3 --normalize_way none"),
        ("0b", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 1.0"),
        ("0c", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0"),
        # merge2
        ("2a", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.0 --decode_latnbest"),
        ("2b", "--pr_local_diff 2.3 --normalize_way none --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.0 --decode_latnbest"),
        ("2c", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_nalpha 1.0"),
        ("2d", "--pr_local_diff 2.3 --normalize_way none --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_nalpha 1.0"),
        ("2e", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 1.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_nalpha 1.0"),
        ("2f", "--pr_local_diff 2.3 --normalize_way none --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 1.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_nalpha 1.0"),
    ]
}

# slightly adding lr for EnDe
TasksEnDE0207_lr = {
    "GPU":6,
    "CMD_RUN": "PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ../../znmt/test.py -v --test_batch_size 1 -o z.%(dataname)s.%(runname)s.bpe -t ../../en_de_data_z5/%(dataname)s.bpe.en ../../en_de_data_z5/%(dataname)s.tok.de -d %(dirname)s/src.v %(dirname)s/trg.v -m %(dirname)s/zbest.model --dynet-devices GPU:6 %(extras)s",
    "CMD_EVAL": "ZMT=../.. bash ../../znmt/scripts/restore.sh <z.%(dataname)s.%(runname)s.bpe | tee z.%(dataname)s.%(runname)s | perl ../..//znmt/scripts/multi-bleu.perl ../../en_de_data_z5/%(dataname)s.tok.de",
    "DATA_NAMES": ["data2014", "data2015", "data2016", "data2012", "data2013"],
    "DIR_NAME": "../../baselines/ed_ev/",
    "CONFS": [
        # basic
        ("0a", "--pr_local_diff 2.3 --normalize_way none"),
        ("0b1", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1"),
        ("0b2", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.2"),
        ("0b3", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.3"),
        ("0b4", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.4"),
        ("0b5", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.5"),
        ("0c", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0"),
        # merge2
        # ("2a", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.0 --decode_latnbest"),
        # ("2b", "--pr_local_diff 2.3 --normalize_way none --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.0 --decode_latnbest"),
        # ("2c", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_nalpha 1.0"),
        # ("2d", "--pr_local_diff 2.3 --normalize_way none --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_nalpha 1.0"),
        # ("2e", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 1.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_nalpha 1.0"),
        # ("2f", "--pr_local_diff 2.3 --normalize_way none --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 1.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_nalpha 1.0"),
        # merge3
        ("3a1", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_lreward 0.0"),
        ("3a2", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_lreward 0.0"),
        ("3b1", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.1 --decode_latnbest --decode_latnbest_lreward 0.1"),
        ("3b2", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.1 --decode_latnbest --decode_latnbest_lreward 0.1"),
        ("3c1", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.2 --decode_latnbest --decode_latnbest_lreward 0.2"),
        ("3c2", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.2 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.2 --decode_latnbest --decode_latnbest_lreward 0.2"),
        ("3d1", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.3 --decode_latnbest --decode_latnbest_lreward 0.3"),
        ("3d2", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.3 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.3 --decode_latnbest --decode_latnbest_lreward 0.3"),
        ("3e1", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.4 --decode_latnbest --decode_latnbest_lreward 0.4"),
        ("3e2", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.4 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.4 --decode_latnbest --decode_latnbest_lreward 0.4"),
    ]
}

# slightly adding lr for EnDe
TasksEnDE0208_lr = {
    "GPU":0,
    "CMD_RUN": "PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ../../znmt/test.py -v --test_batch_size 1 -o z.%(dataname)s.%(runname)s.bpe -t ../../en_de_data_z5/%(dataname)s.bpe.en ../../en_de_data_z5/%(dataname)s.tok.de -d %(dirname)s/src.v %(dirname)s/trg.v -m %(dirname)s/zbest.model --dynet-devices GPU:0 %(extras)s",
    "CMD_EVAL": "ZMT=../.. bash ../../znmt/scripts/restore.sh <z.%(dataname)s.%(runname)s.bpe | tee z.%(dataname)s.%(runname)s | perl ../..//znmt/scripts/multi-bleu.perl ../../en_de_data_z5/%(dataname)s.tok.de",
    "DATA_NAMES": ["data46", "data2014", "data2015", "data2016", "data2012", "data2013"],
    "DIR_NAME": "../../baselines/ed_ev/",
    "CONFS": [
        # basic
        ("0a", "--pr_local_diff 2.3 --normalize_way none"),
        ("0b1", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1"),
        ("0b2", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.2"),
        ("0b3", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.3"),
        ("0b4", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.4"),
        ("0b5", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.5"),
        ("0c", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0"),
        ("3a1", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_lreward 0.0"),
        ("3a2", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_lreward 0.0"),
        ("3b1", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.1 --decode_latnbest --decode_latnbest_lreward 0.1"),
        ("3b2", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.1 --decode_latnbest --decode_latnbest_lreward 0.1"),
        ("3c1", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.2 --decode_latnbest --decode_latnbest_lreward 0.2"),
        ("3c2", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.2 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.2 --decode_latnbest --decode_latnbest_lreward 0.2"),
        ("3d1", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.3 --decode_latnbest --decode_latnbest_lreward 0.3"),
        ("3d2", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.3 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.3 --decode_latnbest --decode_latnbest_lreward 0.3"),
        ("3e1", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.4 --decode_latnbest --decode_latnbest_lreward 0.4"),
        ("3e2", "--pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.4 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.4 --decode_latnbest --decode_latnbest_lreward 0.4"),
    ]
}

# slightly adding lr for EnDe on dev
TasksEnDE0209_lr = {
    "GPU":0,
    "CMD_RUN": "PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ../../znmt/test.py -v --test_batch_size 1 -o z.%(dataname)s.%(runname)s.bpe -t ../../en_de_data_z5/%(dataname)s.bpe.en ../../en_de_data_z5/%(dataname)s.tok.de -d %(dirname)s/src.v %(dirname)s/trg.v -m %(dirname)s/zbest.model --dynet-devices GPU:0 %(extras)s",
    "CMD_EVAL": "ZMT=../.. bash ../../znmt/scripts/restore.sh <z.%(dataname)s.%(runname)s.bpe | tee z.%(dataname)s.%(runname)s | perl ../..//znmt/scripts/multi-bleu.perl ../../en_de_data_z5/%(dataname)s.tok.de",
    "DATA_NAMES": ["data46", "data2014", "data2015", "data2016", "data2012", "data2013"],
    "DIR_NAME": "../../baselines/ed_ev/",
    "CONFS": [
        # norm
        ("a11", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0"),
        ("b11", "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 1.0 --pr_global_lreward 0.0 --decode_latnbest --decode_latnbest_lreward 1.0"),
    ]
}
for i in range(0, 11):
    m = {"a": i/10}
    TasksEnDE0209_lr["CONFS"].append(("a%d"%i, "--pr_local_diff 2.3 --normalize_way add --normalize_alpha %(a)s" % m))
    TasksEnDE0209_lr["CONFS"].append(("b%d"%i, "--pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward %(a)s --decode_latnbest --decode_latnbest_lreward %(a)s" % m))
    TasksEnDE0209_lr["CONFS"].append(("c%d"%i, "--pr_local_diff 2.3 --normalize_way add --normalize_alpha %(a)s --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward %(a)s --decode_latnbest --decode_latnbest_lreward %(a)s" % m))

# -------------------------

TasksZhEn_final = TasksZhEn0207_all

def main():
    task_name = sys.argv[1]
    task = globals()[task_name]
    run(task)
    zhold(task["GPU"])

if __name__ == "__main__":
    main()

# loop
# PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ../../znmt/test.py -v --report_freq 128 --eval_metric ibleu -o loop --loop -d ../z1217_base/{"src","trg"}.v -m ../z1217_base/zbest.model --dynet-devices GPU:7 --beam_size 10 -t a --decode_dump_hiddens
