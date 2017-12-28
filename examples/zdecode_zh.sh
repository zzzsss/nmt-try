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
gpuid=7
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

# PYTHONPATH=$DY_ZROOT/cbuild/python python3.5 -m pdb ../../znmt/test.py -v --report_freq 128 --eval_metric ibleu -o debug -t ../../zh_en_data/Dev-set/nist_2002.{src,ref0} -d ./{"src","trg"}.v -m zbest.model --dynet-devices CPU --decode_output_r2l

# analysis
# PYTHONPATH=$DY_ZROOT/cbuild/python python3 ../../znmt/rerank.py -d ../z1126_3/src.v ../z1126_3/trg.v -m --gold ../../zh_en_data/Dev-set/nist_2002.ref* -t ../../zh_en_data/Dev-set/nist_2002.src ./z.nist_2002.t00.nbest
# python3 ../../znmt/scripts/tools/extract_n.py ../../zh_en_data/Dev-set/nist_2002.* ./z.nist_2002.t00.nbest

# rerank
PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 -m pdb ../../znmt/rerank.py -v --report_freq 128 --eval_metric ibleu -o ./z.nist_2002.debug.rr -t ../../zh_en_data/Dev-set/nist_2002.src ./z.nist_2002.t00.nbest --gold ../../zh_en_data/Dev-set/nist_2002.ref* -d ../z1126_3/{"src","trg"}.v --dynet-devices GPU:7 -m ../z1126_3/{model.e21-u320000,model.e22-u330000,model.e23-u340000} --normalize_way norm --normalize_alpha 1.0

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
