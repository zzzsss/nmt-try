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
# nematus-base
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t nematus --batch_size 80 --patience 3 --extras "dropout_source 0.1 dropout_target 0.1 use_dropout"
# base
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "drop_rec 0.2"
# lr
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "drop_rec 0.2 lrate 0.0002"
# gdrop
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "gdrop_rec 0.2"
# momentum
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "drop_rec 0.2 trainer_type momentum lrate 0.5"
# summ type
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --patience 3 --extras "drop_rec 0.2 summ_type ends"
##===== nematus wmt
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_40000/ -t nematus --batch_size 80 --extras "dropout_source 0.1 dropout_target 0.1 use_dropout"
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_fr_data_40000/ -t nematus --batch_size 80 --extras "dropout_source 0.1 dropout_target 0.1 use_dropout"
