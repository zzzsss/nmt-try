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
# -- adding on 17.10.01
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends idrop_rec 0.2" -p 3
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends idrop_rec 0.2 gdrop_rec 0.2" -p 3
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends idrop_rec 0.2 gdrop_rec 0.2 idrop_embedding 0.1" -p 2
# gdrop-again
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends gdrop_rec 0.2" -p 4
# embed_gdrop
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends idrop_rec 0.2 gdrop_embedding 0.1" -p 5
# embed idrop
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends idrop_rec 0.2 idrop_embedding 0.1" -p 6
# momentum-0.1
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "summ_type ends idrop_rec 0.2 trainer_type momentum lrate 0.1" -p 7
