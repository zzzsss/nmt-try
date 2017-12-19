#!/usr/bin/env bash

# 17.09.25 -- some tryings (mainly on 5048)
mkdir -p z0925_0 z0925_1 z0925_2 z0925_3 z0925_4
# base (dev 31+)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_enc 500"
# dropout (dev 32+)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_enc 500 drop_rec 0.2"
# gdrop (still bad: only 23+)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_enc 500 gdrop_rec 0.2"
# lstm (dev 31.00)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_enc 500 rnn_type lstm"
# lr (dev 32+)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_enc 500 lrate 0.0002"
# small lstm @ x46 (kill early, skip)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_rec 500 hidden_att 500 rnn_type lstm"
# sgd @ x47 (skip)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_enc 500 trainer_type sgd lrate 1.0"
# momentum @ x47 (skip)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_enc 500 trainer_type momentum lrate 0.1"

# 17.09.27 -- going on
# nematus-base (36.55 on dev)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t nematus --batch_size 80 --patience 3 --extras "dropout_source 0.1 dropout_target 0.1 use_dropout"
# base
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "drop_rec 0.2"
# lr (33, early-good, but later ...)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "drop_rec 0.2 lrate 0.0002"
# gdrop (still not good)
# python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "gdrop_rec 0.2"
 python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "drop_rec 0.2 gdrop_rec 0.1"
# momentum (33.75, fluc)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "drop_rec 0.2 trainer_type momentum lrate 0.5"
# summ type
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "drop_rec 0.2 summ_type ends"
##===== nematus wmt
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_40000/ -t nematus --batch_size 80 --extras "dropout_source 0.1 dropout_target 0.1 use_dropout"
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_fr_data_40000/ -t nematus --batch_size 80 --extras "dropout_source 0.1 dropout_target 0.1 use_dropout"
# ------
# find-outs about 170927 on iwslt-en-fr:
# 1. dropout is important, especially gdrop which is still not good in znmt; initial not good, but gradually rise ...
# 1.5. without dropout, it seems that nematus will only reach 30.? (loss 50).
# 2. lr: 0.0002 will top the bleu in 2 or 3 valid points up to 33+, but maybe not better later
# 3. BLUE is related to loss, nematus could get sloss down to 37, but the best of znmt is 45+.
# 4. seems that summ:ends could be better

# 17.09.29
# base (accidentally killed, never mind)
# -- adding on 17.10.01 (killed)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends idrop_rec 0.2" -p 3
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends idrop_rec 0.2 gdrop_rec 0.2" -p 3
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends idrop_rec 0.2 gdrop_rec 0.2 idrop_embedding 0.1" -p 2
# gdrop-again (dev-k1-34.5, test-k5-35.8)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2" -p 4
# embed_gdrop (early kill)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends idrop_rec 0.2 gdrop_embedding 0.1" -p 5
# embed idrop (early kill)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends idrop_rec 0.2 idrop_embedding 0.1" -p 6
# momentum-0.1 (early kill)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends idrop_rec 0.2 trainer_type momentum lrate 0.1" -p 7
# findings:
# without gdrop, it will be hard to get to 34+

# 17.10.02
# base
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1" -p 5
# coverage1
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 coverage_dim 10 coverage_dim_hidden 100" -p 6
# coverage2
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 coverage_dim 100 coverage_dim_hidden 100 att_type biaff" -p 7
## larger dropouts (1003)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.2" -p 2
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.3 idrop_embedding 0.2" -p 3

# 17.10.4 (fix shuffling)
# base (x46) (34.5+, 35.4)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1" -p 6
# diff-gdrop (35.0+, 34.5+)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 gdrop_rec_diff_masks" -p 3
# biaff (35.5+, 35.3+)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 att_type biaff" -p 5
# coverage (35.0+, 35.0)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 coverage_dim 20 coverage_dim_hidden 50" -p 6
# momentum (35.5+, 36.0+)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 trainer_type momentum lrate 0.5 moment 0.6" -p 7
# lstm (x46) (34.0+, 35.2)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 rnn_type lstm" -p 7

# 17.10.8 (tuning again)
# t0-noshuffle (x46) (35.4+, 35.4)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 no_shuffle_training_data" -p 6
# t1-d0.4 (x46) (!!35.7+, 35.7)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.1" -p 7
# idrop (35.4+, 35.5+)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends idrop_rec 0.2 gdrop_rec 0.2 idrop_embedding 0.1" -p 3
# avg (34.3+, 35.5+, not fully)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type avg gdrop_rec 0.2 idrop_embedding 0.1" -p 5
# rnn (33+, 34+)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 dec_type att" -p 6
# lrate (34.8+, 34.9+, not fully)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 lrate 0.0002" -p 7
# momentum (x46) (35.0, 35.1)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 trainer_type momentum lrate 0.2 moment 0.6" -p 3

# findings:
# 1. seems that larger gdrop helps ...
# 2. cGRU helps

# 17.10.11
# 0, base (35.3+, 36.08)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.1" -p 3
# 1, drop1 (36.0, 36.25)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4" -p 5
# 2, drop2 (35.4+, 36.20)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.3 idrop_embedding 0.3 drop_hidden 0.3 drop_embedding 0.3" -p 6
# 3, left for exploring
# ... -p 7
# 4(x46), bi-affine (35.9, 35.96)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.1 att_type biaff" -p 3
# 5(x46), bi-affine + cov (36.0, 35.82)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.1 att_type biaff coverage_dim 10 coverage_dim_hidden 100" -p 6
# 6(x46), bi-affine + cov2 (36.0, 35.93)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.1 att_type biaff coverage_dim 50 coverage_dim_hidden 100" -p 7

# 17.10.15
# now run nematus again on z5 (x46)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t nematus --batch_size 80 --extras "dropout_embedding 0.4 dropout_hidden 0.4 dropout_source 0.4 dropout_target 0.4 use_dropout" -p 6 # (too large dropout, skip)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t nematus --batch_size 80 --extras "dropout_embedding 0.2 dropout_hidden 0.2 dropout_source 0.1 dropout_target 0.1 use_dropout" -p 7 # (36.5, 36.7)
# and back on x48
# 0. base on z5 (36.2, 36.7)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4" -p 3
# 1. base (36.2, 36.3)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4" -p 4
# 2. bi-affine (36.2, 35.8)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4 att_type biaff" -p 5
# 3. small (35.7, 35.5)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4 hidden_rec 500 hidden_att 500" -p 6
# 4. all dropouts (35.6, 36.2)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4 idrop_rec 0.4" -p 7

# findings:
# (maybe) Too-large (many) dropouts are not that good.

# 17.10.20
## now start to turn to z5 & JE
# base nematus (x47) (oom)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../ja_en_data_z5 -t nematus --batch_size 80 --extras "dropout_embedding 0.2 dropout_hidden 0.2 dropout_source 0.1 dropout_target 0.1 n_words_src 50000 n_words 50000 use_dropout" -z 6 -p 6
# 0. drop (35.76, 36.02)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 5
# 1. biaffine (36.21, 36.29)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 att_type biaff" -z 6 -p 7
##
# JE on x46
# 0. drop
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../ja_en_data_z5 -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_thres 50000" -z 6 -p 6
# 1. biaffine
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../ja_en_data_z5 -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 att_type biaff dicts_thres 50000" -z 6 -p 7

# 17.10.26
## final tuning on dropouts
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.5 drop_hidden 0.2 idrop_embedding 0.2 drop_embedding 0.2" -z 6 -p 5
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.5 drop_hidden 0.2 idrop_embedding 0.1 drop_embedding 0.1" -z 6 -p 6
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.5 drop_hidden 0.2 idrop_embedding 0.1 drop_embedding 0.2" -z 6 -p 7
# => 35+, maybe slightly too large gdrop

# 17.10.29
# see what is for wmt_ef (30k vocab)
# python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_fr_data_z5/set2/ -t nematus --batch_size 80 --extras "dropout_source 0.1 dropout_target 0.1 n_words_src 30000 n_words 30000 use_dropout" -p 1
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_fr_data_z5/set2/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_thres 30000" -p 4

# 17.10.30
# ..., don't know what to run, please finish ztry1 as soon as possible
# if several runs will be quite different?
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 5
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 6
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 7
