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
# base (x46)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1" -p 6
# diff-gdrop
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 gdrop_rec_diff_masks" -p 3
# biaff
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 att_type biaff" -p 5
# coverage
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 coverage_dim 20 coverage_dim_hidden 50" -p 6
# momentum
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 trainer_type momentum lrate 0.5 moment 0.6" -p 7
# lstm (x46)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 rnn_type lstm" -p 7
