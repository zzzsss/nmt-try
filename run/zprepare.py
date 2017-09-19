#!/usr/bin/env python3

# preparing for the local shell scripts

import argparse
import re
import os
import sys

def init():
    p = argparse.ArgumentParser()
    p.add_argument("--tool", "-t", type=str, required=True, choices=["nematus", "xnmt", "znmt"])
    p.add_argument("--device", "-p", type=int, required=True, help="-1:cpu >0:gpu")
    p.add_argument("--datadir", "-d", type=str, required=True)
    p.add_argument("--rundir", type=str)    # None means current one (pwd)
    p.add_argument("--src", type=str)   # could be inferred from data_dir name
    p.add_argument("--trg", type=str)   # could be inferred from data_dir name
    p.add_argument("--output", "-o", type=str, default="output.txt")
    # some others
    dicts = {
        # zmt home
        "--zmt": "..",
        # znmt style parameters
        "--max_epochs": 100,
        "--max_updates": 500000,
        "--max_len": 50,
        "--batch_size": 64,
        "--valid_batch_width": 32,
        "--report_freq": 1000,
        "--normalize": 0.0,
        "--dev_beam_size": 1,
        "--valid_freq": 10000,
        "--patience": 5,
        "--anneal_restarts": 2,
        "--test_beam_size": 5
    }
    for k in dicts:
        p.add_argument(k, type=type(dicts[k]), default=dicts[k])
    a = p.parse_args()
    args = vars(a)
    # check them and inferring
    if args["rundir"] is None or args["rundir"] == "":
        bn = os.path.basename(os.path.abspath("."))
        args["rundir"] = "../%s" % bn
    if args["src"] is None or args["trg"] is None:
        # inferring from datadir-name
        LANGS = ["en", "de", "fr", "cn", "ja"]
        fs = re.findall("|".join(LANGS), args["datadir"])
        if len(fs) == 2:
            args["src"], args["trg"] = fs[0], fs[1]
        else:
            raise RuntimeError("Cannot infer langs from datadir %s" % args["datadir"])
    # report
    print("Generating with args as %s" % args)
    return args

def _read_file(f, args, no_env):
    # use env values (this may have some unknown effects --- if not using ${} in the script)
    PATTERN = "\$\{(.*?)\}"
    s = ["#!/usr/bin/env bash"]
    if not no_env:
        for k in args:
            s.append("%s=%s" % (k, args[k]))
    with open(f) as ff:
        ss = ff.read()
        if no_env:
            fs = re.findall(PATTERN, ss)
            lf = re.sub(PATTERN, "%s", ss)
            if len(fs) > 0:
                ss = lf % tuple(args[k] for k in fs)
    return ss

def _get_script(f):
    dname = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dname, f)

def _prepare_file(sinput, output, args, additions):
    no_env = True
    aa = args.copy()
    for k in additions:
        aa[k] = additions[k]
    inp = _get_script(sinput)
    s = _read_file(inp, aa, no_env)
    with open(output, "w") as f:
        f.write(s)

def main():
    args = init()
    is_gpu = (args["device"] >= 0)
    if is_gpu:
        args["gpuid"] = args["device"]
    table = {
        "nematus": [
            ["run_nematus.sh", "_test.sh", args, {"_nematus_device":("cuda" if is_gpu else "cpu"), "_nematus_valid_script":"./_nematus_valid.sh"}],
            ["_nematus_valid.sh", "_nematus_valid.sh", args, {}]
        ],
        "xnmt": [
            ["run_xnmt.sh", "_test.sh", args, {"_xnmt_yaml": "_xnmt.yaml"}],
            ["_xnmt.yaml", "_xnmt.yaml", args, {"_xnmt_valid_every": args["valid_freq"]*args["batch_size"]}]
        ],
        "znmt": [
            ["run_znmt.sh", "_test.sh", args, {}]
        ]
    }
    for eve in table[args["tool"]]:
        _prepare_file(*eve)

if __name__ == "__main__":
    main()

# the test
# python3 ../znmt/run/zprepare.py --valid_freq 100 -d ../en_ja_test -t znmt -p ?
# the run
# python3 ../znmt/run/zprepare.py -d ../data2/en-fr/ -t znmt -p ?
# python3 ../../znmt/run/zpreapre.py --zmt ../.. -d ../data2/en-fr/ -t znmt -p ?
