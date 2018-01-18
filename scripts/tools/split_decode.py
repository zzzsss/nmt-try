#!/bin/python3

# split & decoding: for large decoding

import argparse, os, subprocess, math

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lines', '-l', type=int, default=10000, help="How many lines max for one piece.")
    parser.add_argument('--task', '-t', type=str, default="decode_train", choices=["decode_train"], help="To do which task.")
    parser.add_argument('--restart', '-r', type=int, default=0, help="From which piece to restart.")
    a = parser.parse_args()
    return a

printing = lambda x: print(x, flush=True)

def system(cmd, pp=True, ass=True, popen=True):
    if pp:
        printing("Executing cmd: %s" % cmd)
    if popen:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        n = p.wait()
        output = p.stdout.read()
    else:
        n = os.system(cmd)
        output = None
    if pp:
        printing("Output is: %s" % output)
    if ass:
        assert n==0
    return output

def decode_train(opts):
    train_src = "../../zh_en_data/train.final.zh"
    train_trg = "../../zh_en_data/train.final.en"
    split_src = "./train.zh"
    split_trg = "./train.en"
    #
    lines = opts.lines
    restart_point = max(0, opts.restart)
    # step1: split the training file
    wc_num = int(system("wc -l %s"%train_src).split()[0])
    wc_num0 = int(system("wc -l %s"%train_trg).split()[0])
    assert wc_num == wc_num0
    split_num = math.ceil(wc_num/lines)
    number_pattern = "%0" + str(len(str(split_num))) + "d"
    #
    output_fname = "./output"
    cmd = "PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ../../znmt/test.py --eval_metric ibleu -d ../z1217_base/src.v ../z1217_base/trg.v -m ../z1217_base/zbest.model --dynet-devices GPU:2 --decode_extract_paraf --pr_global_expand 1 --pr_tngram_n 4 --pr_tngram_range 4 --pr_local_diff 2.0 -o ./output.%s -t ./train.zh.%s ./train.en.%s --log log.%s" % tuple([number_pattern]*4)
    get_cmd = lambda n: cmd % tuple([n]*4)
    #
    printing("step1: write to splitted files, number %s, pattern as %s." % (split_num, number_pattern))
    cur_num = 0
    with open(train_src) as fd_src, open(train_trg) as fd_trg:
        continue_flag = True
        while continue_flag:
            cur_lineno = 0
            cur_suffix = "." + number_pattern % cur_num
            printing("-- write to %s" % cur_suffix)
            with open(split_src+cur_suffix, "w") as cur_src_fd, open(split_trg+cur_suffix, "w") as cur_trg_fd:
                while cur_lineno < lines:
                    x = fd_src.readline()
                    y = fd_trg.readline()
                    if len(x)==0 and len(y)==0:
                        continue_flag = False
                        break
                    elif len(x)==0 or len(y)==0:
                        assert False, "Unequal parallel files!!"
                    cur_src_fd.write(x)
                    cur_trg_fd.write(y)
                    cur_lineno += 1
                cur_num += 1
    # step2: running with them
    printing("step2: running with all of them, start from %d." % restart_point)
    for i in range(restart_point, cur_num):
        cur_cmd = get_cmd(i)
        system(cur_cmd)
    # step3: combine
    printing("step3: combine all of them.")
    for pattern in ["./output.%s", "./output.%s.nbest", "./output.%s.nbests",
                    "./output.%s.nbestg", "./output.%s.med.json", "./output.%s.merge.json"]:
        all_file = pattern%"all"
        for i in range(cur_num):
            one_num = number_pattern % i
            one_file = pattern % one_num
            system("cat %s >> %s" % (one_file, all_file))

def main():
    opts = init()
    tasker = {"decode_train": decode_train}[opts.task]
    tasker(opts)

if __name__ == '__main__':
    main()
