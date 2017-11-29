#!/usr/bin/env bash

zmt="../.."
rundir="../z1126_3/"
TEST_SUBDIR="Test-set"
EVAL_SUBDIR="Reference-for-evaluation"
datadir="../../zh_en_data/"
dataname="nist_2002"
output="z"
src="zh"
trg="en"
gpuid=3
#pyargs="-m pdb"
pyargs=""
baseextras=""
function run
{
    echo "running with $1, and with extras $2"
    name=$1
    extras=$2
#    prof_param="-m cProfile -o $name.prof"
    prof_param=""
    PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ${pyargs} ${prof_param} ${zmt}/znmt/test.py -v --report_freq 128 --eval_metric ibleu -o ${output}.$dataname.$1 -t $datadir/$TEST_SUBDIR/$dataname.src $datadir/$EVAL_SUBDIR/$dataname/$dataname.ref0 -d ${rundir}/{"src","trg"}.v -m ${rundir}/zbest.model --dynet-devices GPU:${gpuid} ${baseextras} ${extras}
    perl ${zmt}/znmt/scripts/multi-bleu.perl $datadir/$EVAL_SUBDIR/$dataname/$dataname < ${output}.$dataname.$1
}

# basic & local pruning
run t00 "--test_batch_size 1 --beam_size 1"
run t01 "--test_batch_size 1 --beam_size 5"
run t02 "--test_batch_size 1 --beam_size 10"
run t03 "--test_batch_size 1 --beam_size 20"
run t04 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3"
run t05 "--test_batch_size 1 --beam_size 20 --pr_local_diff 2.3"
run t06 "--test_batch_size 1 --beam_size 10 --pr_local_diff 1.6"
run t07 "--test_batch_size 1 --beam_size 20 --pr_local_diff 1.6"
# normalization
run t10 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0"
run t11 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 0.5"
run t12 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way google --normalize_alpha 1.0"
run t13 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way google --normalize_alpha 0.5"
run t14 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1"
run t15 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.5"
run t16 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way gaussian --normalize_alpha 1.0"
run t17 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way gaussian --normalize_alpha 0.5"

run t2 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1"
run t3 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1 --pr_local_penalty 0.1"
run t4 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1"
run t5 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1 --pr_local_penalty 0.1"
run t6 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1"
run t7 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1 --decode_latnbest"
run t8 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1"
run t9 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1 --decode_latnbest"
