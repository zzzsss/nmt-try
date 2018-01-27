#!/usr/bin/env bash

zmt="../.."
#rundir="../z1126_3/"
#rundir="../z1217_base/"
#rundir="../../baselines/ze_ev/"
rundir="../../baselines/ze_ev_drop/"
TEST_SUBDIR="Test-set"
EVAL_SUBDIR="Reference-for-evaluation"
datadir="../../zh_en_data/"
#dataname="nist_2002"
dataname="nist_36"
output="z"
src="zh"
trg="en"
gpuid=2
pyargs=""
#pyargs="-m pdb"
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

function run_rerank0
{
    echo "running reranking0 with $1, and with extras $2"
    name=$1
    extras=$2
    PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ${pyargs} ${prof_param} ${zmt}/znmt/rerank.py -v --report_freq 128 --eval_metric ibleu -o ${output}.rr.$dataname.$1.rr -t $datadir/$TEST_SUBDIR/$dataname.src ${output}.$dataname.$1.nbest --gold $datadir/$EVAL_SUBDIR/$dataname/$dataname.ref* -d ${rundir}/{"src","trg"}.v -m ${rundir}/zbest.model --dynet-devices GPU:${gpuid} ${baseextras} ${extras}
    perl ${zmt}/znmt/scripts/multi-bleu.perl $datadir/$EVAL_SUBDIR/$dataname/$dataname < ${output}.rr.$dataname.$1.rr
}

# test
# PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 -m pdb ../../znmt/test.py -v --report_freq 128 --eval_metric ibleu -o debug -t ../../zh_en_data/Dev-set/nist_2002.{src,ref0} -d ../z1217_base//{"src","trg"}.v -m ../z1217_base/zbest.model --dynet-devices ?

# paraf
#PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 -m pdb ../../znmt/test.py -v --report_freq 128 --eval_metric ibleu -o debug -d ../z1217_base//{"src","trg"}.v -m ../z1217_base/zbest.model --dynet-devices GPU:2 --decode_extract_paraf --pr_global_expand 1 --pr_tngram_n 4 --pr_tngram_range 4 --pr_local_diff 2.0 -t ../../zh_en_data/Dev-set/nist_2002.{src,ref0,ref1,ref2,ref3}

# analysis
# PYTHONPATH=$DY_ZROOT/cbuild/python python3 ../../znmt/rerank.py -d ../z1217_base/{src,trg}.v -m --gold ../../zh_en_data/Dev-set/nist_2002.ref* -t ../../zh_en_data/Dev-set/nist_2002.src ./z.nist_2002.t00.nbest
# python3 ../../znmt/scripts/tools/extract_n.py ../../zh_en_data/Dev-set/nist_2002.* ./z.nist_2002.t00.nbest

# rerank
# PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 -m pdb ../../znmt/rerank.py -v --report_freq 128 --eval_metric ibleu -o ./z.nist_2002.debug.rr -t ../../zh_en_data/Dev-set/nist_2002.src ./z.nist_2002.t00.nbest --gold ../../zh_en_data/Dev-set/nist_2002.ref* -d ../z1126_3/{"src","trg"}.v --dynet-devices GPU:7 -m ../z1126_3/{model.e21-u320000,model.e22-u330000,model.e23-u340000} --normalize_way norm --normalize_alpha 1.0

function run_rerank
{
    echo "running reranking with $1, and with extras $2"
    name=$1
    extras=$2
    PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ${pyargs} ${prof_param} ${zmt}/znmt/rerank.py -v --report_freq 128 --eval_metric ibleu -o ${output}.rr.$dataname.$1.rr -t $datadir/$TEST_SUBDIR/$dataname.src ${output}.$dataname.$1.nbest --gold $datadir/$EVAL_SUBDIR/$dataname/$dataname.ref* -d ${rundir}/{"src","trg"}.v --dynet-devices GPU:${gpuid} ${baseextras} -m ${rundir}/{model.e21-u320000,model.e22-u330000,model.e23-u340000} ${extras}
    perl ${zmt}/znmt/scripts/multi-bleu.perl $datadir/$EVAL_SUBDIR/$dataname/$dataname < ${output}.rr.$dataname.$1.rr
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
# state-merge
run t20 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5"
run t21 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5"
run t22 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 5"
run t23 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5"
run t24 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5"
run t25 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 4"
run t26 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4"
run t27 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 4"
run t28 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4"
run t29 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 4"
# branching
run t30 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0"
run t31 "--decode_way branch --test_batch_size 1 --beam_size 20 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0"
run t32 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --branching_criterion relative"
run t33 "--decode_way branch --test_batch_size 1 --beam_size 20 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --branching_criterion relative"
run t34 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --branching_expand2"
run t35 "--decode_way branch --test_batch_size 1 --beam_size 20 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --branching_expand2"
run t36 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5"
run t37 "--decode_way branch --test_batch_size 1 --beam_size 20 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5"
run t38 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5"
run t39 "--decode_way branch --test_batch_size 1 --beam_size 20 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5"

run t40 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.3"
run t41 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0"
run t42 "--test_batch_size 1 --beam_size 4 --pr_local_diff 1.6"
run t43 "--test_batch_size 1 --beam_size 4 --pr_local_diff 1.0"
run t44 "--test_batch_size 1 --beam_size 4 --pr_local_diff 0.5"
run t45 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0"
run t46 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0"
run t47 "--test_batch_size 1 --beam_size 4 --pr_local_diff 1.6 --normalize_way norm --normalize_alpha 1.0"
run t48 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 1.0"
run t49 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0"
run t50 "--test_batch_size 1 --beam_size 4 --pr_local_diff 1.6 --normalize_way add --normalize_alpha 1.0"
run t51 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 5"
run t52 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5"
run t53 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5"
run t54 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 6 --pr_tngram_n 5"
run t55 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 7 --pr_tngram_n 5"
run t56 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 3 --pr_tngram_n 5"
run t57 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 4 --pr_tngram_n 5"
run t58 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 5 --pr_tngram_n 5"
run t59 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 6 --pr_tngram_n 5"
run t60 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 7 --pr_tngram_n 5"
run t61 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 4"
run t62 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4"
run t63 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 4"
run t64 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 6 --pr_tngram_n 4"
run t65 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 7 --pr_tngram_n 4"
run t66 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 3 --pr_tngram_n 4"
run t67 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 4 --pr_tngram_n 4"
run t68 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 5 --pr_tngram_n 4"
run t69 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 6 --pr_tngram_n 4"
run t70 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 7 --pr_tngram_n 4"
run t71 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 3"
run t72 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 3"
run t73 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 3"
run t74 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 6 --pr_tngram_n 3"
run t75 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 7 --pr_tngram_n 3"
run t76 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 3 --pr_tngram_n 3"
run t77 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 4 --pr_tngram_n 3"
run t78 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 5 --pr_tngram_n 3"
run t79 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 6 --pr_tngram_n 3"
run t80 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 7 --pr_tngram_n 3"
run t81 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 5"
run t82 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5"
run t83 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5"
run t84 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 6 --pr_tngram_n 5"
run t85 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 7 --pr_tngram_n 5"
run t86 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 3 --pr_tngram_n 5"
run t87 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 4 --pr_tngram_n 5"
run t88 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 5 --pr_tngram_n 5"
run t89 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 6 --pr_tngram_n 5"
run t90 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 2 --pr_tngram_range 7 --pr_tngram_n 5"
run t91 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0"
run t92 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0"
run t93 "--decode_way branch --test_batch_size 1 --beam_size 20 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0"
run t94 "--decode_way branch --test_batch_size 1 --beam_size 30 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0"
run t95 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --branching_criterion b_abs"
run t96 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --branching_criterion b_rel"
run t97 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --branching_criterion rel"
run t98 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --pr_tngram_range 5 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
run t99 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --pr_tngram_range 5 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"

# to analyze with small beam size
run s01 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0"
run s02 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0"
run s03 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0"
run s04 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4"
run s05 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4"
run s06 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_lreward 1.0"
run s07 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_lreward 1.0"
run s08 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s09 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"

run s11 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --penalize_eos 1.0"
run s12 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --penalize_eos 1.0"
run s13 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --penalize_eos 1.0"
run s14 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --penalize_eos 1.0"
run s15 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --penalize_eos 1.0"
run s16 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_lreward 1.0 --penalize_eos 1.0"
run s17 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_lreward 1.0 --penalize_eos 1.0"
run s18 "--test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --penalize_eos 1.0"
run s19 "--decode_way branch --test_batch_size 1 --beam_size 4 --pr_local_diff 2.0 --normalize_way add --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --penalize_eos 1.0"

# to-rerank
for i in `python3 -c '[print("t%02d"%z) for z in range(40)]'`; do
run_rerank $i "--normalize_way norm --normalize_alpha 1.0";
done

#########
# rerun again with norm=1.0
run s20 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0"
run s21 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4"
run s22 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s23 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0"
run s24 "--decode_way branch --test_batch_size 1 --beam_size 10 --branching_fullfill_ratio 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0"
run s25 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4"
run s26 "--decode_way branch --test_batch_size 1 --beam_size 10 --branching_fullfill_ratio 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
# ---- beam=20
run s30 "--test_batch_size 1 --beam_size 20 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0"
run s31 "--test_batch_size 1 --beam_size 20 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4"
run s32 "--test_batch_size 1 --beam_size 20 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s33 "--decode_way branch --test_batch_size 1 --beam_size 20 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0"
run s34 "--decode_way branch --test_batch_size 1 --beam_size 20 --branching_fullfill_ratio 20 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0"
run s35 "--decode_way branch --test_batch_size 1 --beam_size 20 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4"
run s36 "--decode_way branch --test_batch_size 1 --beam_size 20 --branching_fullfill_ratio 20 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"

## rerun again on the new base (z1217_base)
run s01 "--test_batch_size 1 --beam_size 1"
run s02 "--test_batch_size 1 --beam_size 5 --pr_local_diff 2.0"
run s03 "--test_batch_size 1 --beam_size 5 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0"
run s04 "--test_batch_size 1 --beam_size 5 --pr_local_diff 2.0 --no_model_softmax"
run s05 "--test_batch_size 1 --beam_size 5 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --no_model_softmax"
run s06 "--test_batch_size 1 --beam_size 5 --pr_local_diff 2.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4"
run s07 "--test_batch_size 1 --beam_size 5 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4"
run s08 "--test_batch_size 1 --beam_size 5 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s09 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.0"
run s10 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0"
run s11 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --no_model_softmax"
run s12 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --no_model_softmax"
run s13 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2. --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4"
run s14 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4"
run s15 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.0 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"

## rerun again with ze_ev
run s00 "--beam_size 1"
run s01 "--beam_size 5 --pr_local_diff 2.0 --normalize_alpha 0.0"
run s02 "--beam_size 5 --pr_local_diff 2.0 --normalize_alpha 1.0"
run s03 "--beam_size 5 --pr_local_diff 2.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 3"
run s04 "--beam_size 5 --pr_local_diff 2.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 3 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s05 "--beam_size 5 --pr_local_diff 2.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4"
run s06 "--beam_size 5 --pr_local_diff 2.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s07 "--beam_size 5 --pr_local_diff 2.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5"
run s08 "--beam_size 5 --pr_local_diff 2.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s09 "--beam_size 10 --pr_local_diff 2.0 --normalize_alpha 0.0"
run s10 "--beam_size 10 --pr_local_diff 2.0 --normalize_alpha 1.0"
run s11 "--beam_size 10 --pr_local_diff 2.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 3"
run s12 "--beam_size 10 --pr_local_diff 2.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 3 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s13 "--beam_size 10 --pr_local_diff 2.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4"
run s14 "--beam_size 10 --pr_local_diff 2.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s15 "--beam_size 10 --pr_local_diff 2.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5"
run s16 "--beam_size 10 --pr_local_diff 2.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"

#run s20 "--beam_size 10 --pr_local_diff 2.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5 --decode_dump_hiddens"
#
# rerun, diff-1.0 is bad ...
run check_sg "--beam_size 3 --pr_local_diff 2.0 --normalize_alpha 1.0 --decode_dump_hiddens"
run s21 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s22 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s23 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s24 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s25 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s26 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 6 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s27 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 7 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s28 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s29 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s30 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s31 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s32 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s33 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 6 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s34 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 7 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
#
run s35 "--beam_size 10 --pr_local_diff 3.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s36 "--beam_size 10 --pr_local_diff 3.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s37 "--beam_size 10 --pr_local_diff 3.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s38 "--beam_size 10 --pr_local_diff 3.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s39 "--beam_size 10 --pr_local_diff 3.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s40 "--beam_size 10 --pr_local_diff 3.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 6 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s41 "--beam_size 10 --pr_local_diff 3.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 7 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s42 "--beam_size 10 --pr_local_diff 3.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s43 "--beam_size 10 --pr_local_diff 3.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s44 "--beam_size 10 --pr_local_diff 3.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s45 "--beam_size 10 --pr_local_diff 3.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s46 "--beam_size 10 --pr_local_diff 3.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s47 "--beam_size 10 --pr_local_diff 3.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 6 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"
run s48 "--beam_size 10 --pr_local_diff 3.0 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 7 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0"

run s50 "--beam_size 2 --pr_local_diff 2.3 --normalize_alpha 1.0 --decode_dump_hiddens"
run s51 "--beam_size 3 --pr_local_diff 2.3 --normalize_alpha 1.0 --decode_dump_hiddens"
run s52 "--beam_size 5 --pr_local_diff 2.3 --normalize_alpha 1.0 --decode_dump_hiddens"
run s53 "--beam_size 8 --pr_local_diff 2.3 --normalize_alpha 1.0 --decode_dump_hiddens"
run s54 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --decode_dump_hiddens"
run s55 "--beam_size 12 --pr_local_diff 2.3 --normalize_alpha 1.0 --decode_dump_hiddens"
run s56 "--beam_size 16 --pr_local_diff 2.3 --normalize_alpha 1.0 --decode_dump_hiddens"
run s57 "--beam_size 20 --pr_local_diff 2.3 --normalize_alpha 1.0 --decode_dump_hiddens"

run debug "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --cov_record_mode max"

# 18.01.29
# for all beam sizes
for i in `python3 -c '[print(z+1) for z in range(30)]'`; do
    echo run zo$i "--beam_size $i --normalize_alpha 1.0";
    run zo$i "--beam_size $i --normalize_alpha 1.0";
    echo run zn$i "--beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0";
    run zn$i "--beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0";
    echo run zm$i "--beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
    run zm$i "--beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
done
run zz1 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --cov_record_mode max --cov_l1_thresh 0.1 --cov_upper_bound 1"
run zz2 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --cov_record_mode max --cov_l1_thresh 0.1 --cov_upper_bound 2"
run zz3 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --cov_record_mode sum --cov_l1_thresh 0.1 --cov_upper_bound 1"
run zz4 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --cov_record_mode sum --cov_l1_thresh 0.1 --cov_upper_bound 2"
run zz5 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --cov_record_mode sum --cov_l1_thresh 0.1 --cov_upper_bound 1 --cov_average"
run zz6 "--beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --cov_record_mode sum --cov_l1_thresh 0.1 --cov_upper_bound 2 --cov_average"

# 18.01.30
for i in `python3 -c '[print(z+1) for z in range(1, 20)]'`; do
    echo run y0m$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0";
    run y0m$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0";
    echo run y1m$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
    run y1m$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
    echo run y2m$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
    run y2m$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
    echo run y3m$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
    run y3m$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
    echo run y4m$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
    run y4m$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
done

# 18.01.31
for i in `python3 -c '[print(z+1) for z in range(1, 20)];print(25);print(30)'`; do
    #
    run x0a-$i "--test_batch_size 4 --beam_size $i --normalize_way none";
    run x0b-$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_way none";
    run x0c-$i "--test_batch_size 4 --beam_size $i --normalize_alpha 1.0";
    run x0d-$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0";
    #
    run x1a-$i "--test_batch_size 4 --beam_size $i --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --normalize_way none";
    run x1b-$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --normalize_way none";
    run x1c-$i "--test_batch_size 4 --beam_size $i --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
    run x1d-$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
    #
    run x2a-$i "--test_batch_size 4 --beam_size $i --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5 --normalize_way none";
    run x2b-$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5 --normalize_way none";
    run x2c-$i "--test_batch_size 4 --beam_size $i --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0";
    run x2d-$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0";
    #
    run x3a-$i "--test_batch_size 4 --beam_size $i --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5 --normalize_way none";
    run x4b-$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5 --normalize_way none";
    run x5c-$i "--test_batch_size 4 --beam_size $i --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0";
    run x6d-$i "--test_batch_size 4 --beam_size $i --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5 --decode_latnbest --decode_latnbest_nalpha 1.0";
done

# 18.02.05: sdec0205
run z0 "--test_batch_size 8 --beam_size 10 --normalize_way none";
run z1 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way none";
run z2 "--test_batch_size 8 --beam_size 10 --normalize_alpha 1.0";
run z3 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0";
#
run za0 "--test_batch_size 8 --beam_size 10 --normalize_way add --normalize_alpha 0.5";
run za1 "--test_batch_size 8 --beam_size 10 --normalize_way add --normalize_alpha 1.0";
run za2 "--test_batch_size 8 --beam_size 10 --normalize_way add --normalize_alpha 1.5";
run za3 "--test_batch_size 8 --beam_size 10 --normalize_way add --normalize_alpha 2.0";
run zn0 "--test_batch_size 8 --beam_size 10 --normalize_way norm --normalize_alpha 0.5";
run zn1 "--test_batch_size 8 --beam_size 10 --normalize_way norm --normalize_alpha 1.0";
run zn2 "--test_batch_size 8 --beam_size 10 --normalize_way norm --normalize_alpha 1.5";
run zn3 "--test_batch_size 8 --beam_size 10 --normalize_way norm --normalize_alpha 2.0";
run zp0 "--test_batch_size 8 --beam_size 10 --normalize_way none --penalize_eos 0.5";
run zp1 "--test_batch_size 8 --beam_size 10 --normalize_way none --penalize_eos 1.0";
run zp2 "--test_batch_size 8 --beam_size 10 --normalize_way none --penalize_eos 1.5";
run zp3 "--test_batch_size 8 --beam_size 10 --normalize_way none --penalize_eos 2.0";
run zda0 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.5";
run zda1 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 1.0";
run zda2 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 1.5";
run zda3 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 2.0";
run zdn0 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 0.5";
run zdn1 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0";
run zdn2 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.5";
run zdn3 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 2.0";
run zdp0 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way none --penalize_eos 0.5";
run zdp1 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way none --penalize_eos 1.0";
run zdp2 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way none --penalize_eos 1.5";
run zdp3 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way none --penalize_eos 2.0";
#
run z4 "--test_batch_size 8 --beam_size 10 --normalize_way none --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4";
run z5 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way none --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4";
run z6 "--test_batch_size 8 --beam_size 10 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4";
run z7 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4";
run z8 "--test_batch_size 8 --beam_size 10 --normalize_way none --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest";
run z9 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way none --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest";
run z10 "--test_batch_size 8 --beam_size 10 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
run z11 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
run_rerank0 z4
run_rerank0 z5
run_rerank0 z6
run_rerank0 z7
run_rerank0 z8
run_rerank0 z9
run_rerank0 z10
run_rerank0 z11

# sdec0205_2
#
run d01 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 1.0";
run d02 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 1.1";
run d03 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 1.2";
run d04 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 1.3";
run d05 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 1.4";
run d06 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 1.5";
#
run d11 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0";
run d12 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.1";
run d13 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.2";
run d14 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.3";
run d15 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.4";
run d16 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.5";
#
run d21 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0";
run d22 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --hid_sim_metric cos --hid_sim_thresh 0.1";
run d23 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --hid_sim_metric cos --hid_sim_thresh 0.01";
run d24 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --hid_sim_metric cos --hid_sim_thresh 0.005";
run d25 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --merge_diff_metric med --merge_diff_thresh 2";
run d26 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --merge_diff_metric med --merge_diff_thresh 5";
run d27 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --decode_latnbest --decode_latnbest_nalpha 1.0 --merge_diff_metric med --merge_diff_thresh 10";
#
run d31 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 1.0 --decode_latnbest --decode_latnbest_lreward 1.0";
run d32 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 1.0 --decode_latnbest --decode_latnbest_lreward 1.0 --hid_sim_metric cos --hid_sim_thresh 0.1";
run d33 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 1.0 --decode_latnbest --decode_latnbest_lreward 1.0 --hid_sim_metric cos --hid_sim_thresh 0.01";
run d34 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 1.0 --decode_latnbest --decode_latnbest_lreward 1.0 --hid_sim_metric cos --hid_sim_thresh 0.005";
run d35 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 1.0 --decode_latnbest --decode_latnbest_lreward 1.0 --merge_diff_metric med --merge_diff_thresh 2";
run d36 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 1.0 --decode_latnbest --decode_latnbest_lreward 1.0 --merge_diff_metric med --merge_diff_thresh 5";
run d37 "--test_batch_size 8 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 1.0 --decode_latnbest --decode_latnbest_lreward 1.0 --merge_diff_metric med --merge_diff_thresh 10";
