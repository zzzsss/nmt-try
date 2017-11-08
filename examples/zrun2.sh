#!/usr/bin/env bash

## 17.11.08 rerun the new ones
#python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --test_beam_size 1 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 4
#python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --test_beam_size 1 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 train_scale 1.0" -z 6 -p 5
#python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --test_beam_size 1 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 bk_init_nl gaussian" -z 6 -p 6
#python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --test_beam_size 1 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 train_scale 1.0 bk_init_nl gaussian" -z 6 -p 7
#
## on x46
#python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --test_beam_size 1 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 bk_init_nl gaussian" -z 6 -p 0
#python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --test_beam_size 1 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 drop_hidden 0.2 drop_embedding 0.1 bk_init_nl gaussian" -z 6 -p 1
#python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --test_beam_size 1 --extras "summ_type ends gdrop_rec 0.2 idrop_embedding 0.1 drop_hidden 0.2 drop_embedding 0.1 bk_init_nl gaussian rnn_type gru2" -z 6 -p 7

# renew
# 17.11.09
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --patience 3 --extras "gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 1
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --patience 3 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 4
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --patience 3 --extras "gdrop_rec 0.2 idrop_embedding 0.1 drop_hidden 0.2 drop_embedding 0.1" -z 6 -p 5
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --patience 3 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 bk_init_nl gaussian" -z 6 -p 6
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --patience 3 --extras "gdrop_rec 0.2 idrop_embedding 0.1 drop_hidden 0.2 drop_embedding 0.1 bk_init_nl gaussian" -z 6 -p 7
# x46
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --patience 3 --extras "gdrop_rec 0.3 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 4
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --patience 3 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 bk_init_nl random" -z 6 -p 7
