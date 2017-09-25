#!/usr/bin/env bash

# 17.09.25 -- some tryings (mainly on 5048)
mkdir -p z0925_0 z0925_1 z0925_2 z0925_3 z0925_4
# default
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_enc 500"
# dropout
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_enc 500 drop_rec 0.2"
# gdrop
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_enc 500 gdrop_rec 0.2"
# lstm
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_enc 500 rnn_type lstm"
# lr
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_enc 500 lrate 0.0002"
# small lstm (x46)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_rec 500 hidden_att 500 rnn_type lstm"
# sgd (x47)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_enc 500 trainer_type sgd lrate 1.0"
# momentum (x47)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/en-fr/ -t znmt --batch_size 80 --extras "hidden_enc 500 trainer_type momentum lrate 0.1"
