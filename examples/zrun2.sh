#!/usr/bin/env bash

# renew
# 17.11.12
## on x48
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 4
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 5
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.1 drop_hidden 0.2 drop_embedding 0.1" -z 6 -p 6
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4" -z 6 -p 7
#
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 hidden_att 500" -z 6 -p 2
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 rnn_type gru2" -z 6 -p 1
## on x46
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 train_len_uidx 100000 train_len_xadd" -z 6 -p 4
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 train_len_uidx 100000 train_len_xadd ZZ train_len_xback" -z 6 -p 7
