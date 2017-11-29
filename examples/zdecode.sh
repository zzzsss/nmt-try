#!/usr/bin/env bash

zmt="../.."
rundir="."
datadir="../../data2/wit3-en-fr_z5/"
output="z"
src="en"
trg="fr"
gpuid=3
#pyargs="-m pdb"
pyargs=""
baseextras="--dim_word 500 --hidden_att 500 --rnn_type gru2"
function run
{
    echo "running with $1, and with extras $2"
    name=$1
    extras=$2
#    prof_param="-m cProfile -o $name.prof"
    prof_param=""
    PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ${pyargs} ${prof_param} ${zmt}/znmt/test.py -v --report_freq 128 -o ${output}.$1 -t ${datadir}/test.final.{${src},${trg}.restore} -d ${rundir}/{"src","trg"}.v -m ${rundir}/zbest.model --dynet-devices GPU:${gpuid} ${baseextras} ${extras}
    ZMT=${zmt} bash ${zmt}/znmt/scripts/restore.sh <${output}.$1 | perl ${zmt}/znmt/scripts/multi-bleu.perl ${datadir}/test.final.${trg}.restore
}

### based on z1117_0 ##
## basic one         (BLEU = 36.41, 64.4/42.3/30.0/21.5 (BP=1.000, ratio=1.000, hyp_len=25251, ref_len=25263)
#run 0basic "--beam_size 10"
## !! maybe in speakings, the sentences are already short and brief, thus longer length (normalizing) is not good?
## norm 1.0          (BLEU = 36.14, 63.6/41.8/29.8/21.5 (BP=1.000, ratio=1.023, hyp_len=25853, ref_len=25263)
#run 1norm "--beam_size 10 --normalize_way norm --normalize_alpha 1.0"
## norm google 1.0   (BLEU = 36.29, 63.9/42.0/30.0/21.6 (BP=1.000, ratio=1.017, hyp_len=25698, ref_len=25263)
#run 2google "--beam_size 10 --normalize_way google --normalize_alpha 1.0"
## penalty 1.0       (BLEU = 35.43, 62.3/41.0/29.3/21.1 (BP=1.000, ratio=1.057, hyp_len=26700, ref_len=25263)
#run 3add "--beam_size 10 --normalize_way add --normalize_alpha 1.0"
## gaussian 1.0      (BLEU = 35.12, 62.7/40.8/28.9/20.6 (BP=1.000, ratio=1.027, hyp_len=25936, ref_len=25263)
#run 4gaussian "--beam_size 10 --normalize_way gaussian --normalize_alpha 1.0"
## ---
## prune length      (BLEU = 36.20, 64.0/42.0/29.8/21.4 (BP=1.000, ratio=1.006, hyp_len=25415, ref_len=25263)
#run p00 "--beam_size 10 --pr_len_klow 1.0 --pr_len_khigh 1.0"
## prune local       (BLEU = 36.15, 64.1/42.0/29.8/21.3 (BP=1.000, ratio=1.000, hyp_len=25258, ref_len=25263)
#run p01 "--beam_size 10 --pr_local_expand 4 --pr_local_diff 1.0 --pr_local_penalty 0.1"
## --
#run p10 "--beam_size 10 --pr_len_klow 2.0 --pr_len_khigh 2.0"
#run p11 "--beam_size 10 --pr_len_klow 3.0 --pr_len_khigh 3.0"
#run p12 "--beam_size 10 --test_batch_size 1"
#run p13 "--beam_size 20 --test_batch_size 1"
#run p14 "--beam_size 40 --test_batch_size 1"
#run p15 "--beam_size 100 --test_batch_size 1"
## ngram
#run p02 "--beam_size 10 --pr_local_expand 4 --pr_local_diff 1.0 --pr_local_penalty 0.1 --pr_global_expand 2 --pr_tngram_range 1"
#run p03 "--beam_size 10 --pr_local_expand 4 --pr_local_diff 1.0 --pr_local_penalty 0.1 --pr_global_expand 2 --pr_tngram_range 2"
#run p04 "--beam_size 10 --pr_local_expand 4 --pr_local_diff 1.0 --pr_local_penalty 0.1 --pr_global_expand 2 --pr_tngram_range 3"
## lattice
#run p02 "--beam_size 10 --pr_local_expand 4 --pr_local_diff 1.0 --pr_local_penalty 0.1 --pr_global_expand 2 --pr_tngram_range 1 --decode_latnbest"

## another wave of running (z1117_0) ##
# -- basic
run r00 "--test_batch_size 1 --beam_size 1"
run r01 "--test_batch_size 1 --beam_size 2"
run r02 "--test_batch_size 1 --beam_size 4"
run r03 "--test_batch_size 1 --beam_size 8"
run r04 "--test_batch_size 1 --beam_size 10"
# -- normalize
run r05 "--test_batch_size 1 --beam_size 10 --normalize_way norm --normalize_alpha 1.0"
run r06 "--test_batch_size 1 --beam_size 10 --normalize_way norm --normalize_alpha 0.5"
run r07 "--test_batch_size 1 --beam_size 10 --normalize_way google --normalize_alpha 1.0"
run r08 "--test_batch_size 1 --beam_size 10 --normalize_way google --normalize_alpha 0.5"
run r09 "--test_batch_size 1 --beam_size 10 --normalize_way add --normalize_alpha 0.1"
run r10 "--test_batch_size 1 --beam_size 10 --normalize_way add --normalize_alpha 0.5"
run r11 "--test_batch_size 1 --beam_size 10 --normalize_way gaussian --normalize_alpha 1.0"
run r12 "--test_batch_size 1 --beam_size 10 --normalize_way gaussian --normalize_alpha 0.5"
# -- local prune & penalty
run r13 "--test_batch_size 1 --beam_size 10 --pr_local_expand 2"
run r14 "--test_batch_size 1 --beam_size 10 --pr_local_expand 3"
run r15 "--test_batch_size 1 --beam_size 10 --pr_local_expand 4"
run r16 "--test_batch_size 1 --beam_size 10 --pr_local_expand 5"
run r17 "--test_batch_size 1 --beam_size 10 --pr_local_diff 0.9"    # log(0.4)
run r18 "--test_batch_size 1 --beam_size 10 --pr_local_diff 1.6"    # log(0.2)
run r19 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3"    # log(0.1)
run r20 "--test_batch_size 1 --beam_size 10 --pr_local_penalty 0.1"
run r21 "--test_batch_size 1 --beam_size 10 --pr_local_penalty 0.5"
run r22 "--test_batch_size 1 --beam_size 10 --pr_local_penalty 1.0"
run r23 "--test_batch_size 1 --beam_size 10 --pr_local_expand 2 --pr_local_diff 0.9 --pr_local_penalty 0.1"
run r24 "--test_batch_size 1 --beam_size 10 --pr_local_expand 2 --pr_local_diff 0.9 --pr_local_penalty 0.5"
run r25 "--test_batch_size 1 --beam_size 10 --pr_local_expand 2 --pr_local_diff 0.9 --pr_local_penalty 1.0"
run r26 "--test_batch_size 1 --beam_size 10 --pr_local_expand 5 --pr_local_diff 2.3 --pr_local_penalty 0.1"
run r27 "--test_batch_size 1 --beam_size 10 --pr_local_expand 5 --pr_local_diff 2.3 --pr_local_penalty 0.5"
run r28 "--test_batch_size 1 --beam_size 10 --pr_local_expand 5 --pr_local_diff 2.3 --pr_local_penalty 1.0"
run r29 "--test_batch_size 1 --beam_size 10 --pr_local_expand 2 --pr_local_diff 0.9"
run r30 "--test_batch_size 1 --beam_size 10 --pr_local_expand 5 --pr_local_diff 0.9"
run r31 "--test_batch_size 1 --beam_size 10 --pr_local_expand 2 --pr_local_diff 1.6"
run r32 "--test_batch_size 1 --beam_size 10 --pr_local_expand 5 --pr_local_diff 1.6"
# -- ngram pruning
run r33 "--test_batch_size 1 --beam_size 10 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 3"
run r34 "--test_batch_size 1 --beam_size 10 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 3"
run r35 "--test_batch_size 1 --beam_size 10 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 3"
run r36 "--test_batch_size 1 --beam_size 10 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 3"
run r37 "--test_batch_size 1 --beam_size 10 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5"
run r38 "--test_batch_size 1 --beam_size 10 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5"
run r39 "--test_batch_size 1 --beam_size 10 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 5"
run r40 "--test_batch_size 1 --beam_size 10 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5"
run r41 "--test_batch_size 1 --beam_size 10 --pr_global_expand 2 --pr_tngram_range 1 --pr_tngram_n 3"
run r42 "--test_batch_size 1 --beam_size 10 --pr_global_expand 2 --pr_tngram_range 2 --pr_tngram_n 3"
run r43 "--test_batch_size 1 --beam_size 10 --pr_global_expand 2 --pr_tngram_range 3 --pr_tngram_n 3"
run r44 "--test_batch_size 1 --beam_size 10 --pr_global_expand 2 --pr_tngram_range 4 --pr_tngram_n 3"
run r45 "--test_batch_size 1 --beam_size 10 --pr_global_expand 2 --pr_tngram_range 1 --pr_tngram_n 5"
run r46 "--test_batch_size 1 --beam_size 10 --pr_global_expand 2 --pr_tngram_range 2 --pr_tngram_n 5"
run r47 "--test_batch_size 1 --beam_size 10 --pr_global_expand 2 --pr_tngram_range 3 --pr_tngram_n 5"
run r48 "--test_batch_size 1 --beam_size 10 --pr_global_expand 2 --pr_tngram_range 4 --pr_tngram_n 5"
# -- larger beam size + normalize/pruning
run r49 "--test_batch_size 1 --beam_size 50"
run r50 "--test_batch_size 1 --beam_size 50 --normalize_way norm --normalize_alpha 1.0"
run r51 "--test_batch_size 1 --beam_size 50 --normalize_way gaussian --normalize_alpha 1.0"
run r52 "--test_batch_size 1 --beam_size 50 --pr_local_expand 5 --pr_local_diff 0.9"
run r53 "--test_batch_size 1 --beam_size 50 --normalize_way norm --normalize_alpha 1.0 --pr_local_expand 5 --pr_local_diff 0.9"
run r54 "--test_batch_size 1 --beam_size 50 --normalize_way gaussian --normalize_alpha 1.0 --pr_local_expand 5 --pr_local_diff 0.9"
run r55 "--test_batch_size 1 --beam_size 50 --pr_local_expand 5 --pr_local_diff 0.9 --pr_local_penalty 1.0"
run r56 "--test_batch_size 1 --beam_size 50 --pr_local_expand 5 --pr_local_diff 0.9 --pr_local_penalty 1.0 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5"
run r57 "--test_batch_size 1 --beam_size 50 --pr_local_expand 5 --pr_local_diff 0.9 --pr_local_penalty 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5"
run r58 "--test_batch_size 1 --beam_size 50 --pr_local_expand 5 --pr_local_diff 0.9 --pr_local_penalty 1.0 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 5"
run r59 "--test_batch_size 1 --beam_size 50 --pr_local_expand 5 --pr_local_diff 0.9 --pr_local_penalty 1.0 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5"
# -----
# findings
# 1. greedy is not enough even for basic ones without norm:
# -> 35.09, 35.76, 36.19, 36.36, 36.39, 36.25 (1,2,4,8,10,50<10>)
# 2. norm for this dataset is not obvious
# 3. local-pruning (expand not obvious, diff maybe should not be too small, penalty??(bug??))
# 4. ngram-pruning (?, has not enable lattice yet, wait to see)
# 5. larger (??)
# => summarize: pr_local_diff 2.3/1.6, slight norm, ngram wait to see

## another wave of running V2 (z1117_0) ##
# beam size
run r60 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3"
run r61 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_local_penalty 0.1"
run r62 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_local_penalty 0.5"
run r63 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_local_penalty 1.0"
run r64 "--test_batch_size 1 --beam_size 20 --pr_local_diff 2.3"
run r65 "--test_batch_size 1 --beam_size 40 --pr_local_diff 2.3"
run r66 "--test_batch_size 1 --beam_size 80 --pr_local_diff 2.3"
run r67 "--test_batch_size 1 --beam_size 20"
run r68 "--test_batch_size 1 --beam_size 40"
run r69 "--test_batch_size 1 --beam_size 80"
# ngram
run r70 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5"
run r71 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5"
run r72 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 5"
run r73 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5"
run r74 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5"
run r75 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5"
run r76 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5"
run r77 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 5"
run r78 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5"
run r79 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5"
run r80 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3"

# lattice
run debug "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3"
# -> BLEU = 36.39, 64.4/42.3/30.0/21.6 (BP=0.998, ratio=0.998, hyp_len=25224, ref_len=25263)
run debug "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5"
# -> BLEU = 36.29, 64.9/42.6/30.3/21.8 (BP=0.987, ratio=0.987, hyp_len=24941, ref_len=25263)
run debug "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5"
run debug "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5 --decode_latnbest"

# rerun them again
run s01 "--test_batch_size 1 --beam_size 1"
run s02 "--test_batch_size 1 --beam_size 5"
run s03 "--test_batch_size 1 --beam_size 10"
run s04 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3"
run s05 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_local_penalty 0.1"
run s06 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_local_penalty 0.5"
run s07 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_local_penalty 1.0"
run s08 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 0.5"
run s09 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1"
run s10 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way gaussian --normalize_alpha 0.5"
run s11 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5"
run s12 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 2 --pr_tngram_range 5 --pr_tngram_n 5"
run s13 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 6"
run s14 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 2 --pr_tngram_range 5 --pr_tngram_n 6"
run s15 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 7"
run s16 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 2 --pr_tngram_range 5 --pr_tngram_n 7"
run s17 "--decode_way branch --test_batch_size 1 --beam_size 10"
run s18 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3"
run s19 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5"
run s20 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 2 --pr_tngram_range 5 --pr_tngram_n 5"
run s21 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 6"
run s22 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 2 --pr_tngram_range 5 --pr_tngram_n 6"
run s23 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 7"
run s24 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 2 --pr_tngram_range 5 --pr_tngram_n 7"
run s25 "--decode_way branch --test_batch_size 1 --beam_size 20 --pr_local_diff 2.3"
run s26 "--decode_way branch --test_batch_size 1 --beam_size 30 --pr_local_diff 2.3"
run s27 "--decode_way branch --test_batch_size 1 --beam_size 20 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1"
run s28 "--decode_way branch --test_batch_size 1 --beam_size 30 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1"
run s29 "--decode_way branch --test_batch_size 1 --beam_size 20 --pr_local_diff 2.3 --normalize_way gaussian --normalize_alpha 0.5"
run s30 "--decode_way branch --test_batch_size 1 --beam_size 30 --pr_local_diff 2.3 --normalize_way gaussian --normalize_alpha 0.5"

# 17.12.02
run s31 "--test_batch_size 1 --beam_size 20 --pr_local_diff 2.3 --pr_local_penalty 0.1 --normalize_way add --normalize_alpha 0.1"
run s32 "--test_batch_size 1 --beam_size 30 --pr_local_diff 2.3 --pr_local_penalty 0.1 --normalize_way add --normalize_alpha 0.1"
run s33 "--test_batch_size 1 --beam_size 40 --pr_local_diff 2.3 --pr_local_penalty 0.1 --normalize_way add --normalize_alpha 0.1"
run s34 "--decode_way branch --test_batch_size 1 --beam_size 20 --pr_local_diff 2.3 --pr_local_penalty 0.1 --normalize_way add --normalize_alpha 0.1"
run s35 "--decode_way branch --test_batch_size 1 --beam_size 30 --pr_local_diff 2.3 --pr_local_penalty 0.1 --normalize_way add --normalize_alpha 0.1"
run s36 "--decode_way branch --test_batch_size 1 --beam_size 40 --pr_local_diff 2.3 --pr_local_penalty 0.1 --normalize_way add --normalize_alpha 0.1"
run s37 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1"
run s38 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1"
run s39 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1"
run s40 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1"
run s41 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1"
run s42 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1 --decode_latnbest"
run s43 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1 --decode_latnbest"
run s44 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1 --decode_latnbest"
run s45 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1 --decode_latnbest"
run s46 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1 --decode_latnbest"
run s47 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1"
run s48 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1"
run s49 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1"
run s50 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1"
run s51 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1"
run s52 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1 --decode_latnbest"
run s53 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1 --decode_latnbest"
run s54 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 3 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1 --decode_latnbest"
run s55 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 4 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1 --decode_latnbest"
run s56 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 5 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1 --decode_latnbest"

# python3 ../../znmt/scripts/tools/extract_n.py ../../data2/wit3-en-fr_z5/test.final.{en,fr} ./z.s09.nbests 102

# 17.12.04
# specifically rerun it
run t0 "--test_batch_size 1 --beam_size 1"
run t1 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3"
run t2 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1"
run t3 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1 --pr_local_penalty 0.1"
run t4 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1"
run t5 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.1 --pr_local_penalty 0.1"
run t6 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1"
run t7 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1 --decode_latnbest"
run t8 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1"
run t9 "--decode_way branch --test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --pr_global_expand 1 --pr_tngram_range 1 --pr_tngram_n 5 --normalize_way add --normalize_alpha 0.1 --decode_latnbest"
