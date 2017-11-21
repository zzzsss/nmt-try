#!/usr/bin/env bash

# renew
# 17.11.12
## on x48 (all around 35 on dev)
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
# -> results
# -- most of them are not good, max only 35.5+ on dev of drop-all-0.4 or hidden-att-500

# 17.11.16
## re-tuning on the new set
# todo(warn): all base on hidden_att->500, dim_word->500, rnn_type->gru2
# todo(warn): this is based on maybe-not-good init of lookup-glorot
#python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4" -z 6 -p 4
#python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4 bk_init_enabled" -z 6 -p 5
#python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 bk_init_enabled" -z 6 -p 6
#python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.0 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.0 bk_init_enabled" -z 6 -p 7
#python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 bk_init_enabled ZZ bk_init_nl random" -z 6 -p 1
#python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 bk_init_enabled ZZ bk_init_nl gaussian" -z 6 -p 2
# todo(warn) giving up the above
# x46
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4 train_len_uidx 100000 train_len_xadd ZZ train_len_lambda 0.1" -z 6 -p 4
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4 train_len_uidx 100000 train_len_xadd ZZ train_len_xback ZZ train_len_lambda 0.1" -z 6 -p 7

# 17.11.17
# (35.93, 36.19)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4" -z 6 -p 2
# (35.41, 35.43)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 3
# (35.42, 35.58)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 bk_init_enabled ZZ" -z 6 -p 4
# (35.29, 35.52)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 bk_init_enabled ZZ" -z 6 -p 5
# (34.84, 35.52)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.2 idrop_embedding 0.1 drop_hidden 0.2 drop_embedding 0.1 bk_init_enabled ZZ" -z 6 -p 6
# (31+, 32+)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.0 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.0 bk_init_enabled ZZ" -z 6 -p 7
# --
# seems like that all-drop-0.4 is all right for old inits

# 17.11.20 (run basic ones for EN_DE,JA-EN,ZH-EN)
# x46
# wit3-en-de: (24.51, 25.36)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-de_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4" -z 6 -p 4
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4" -z 6 -p 7
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 4
# x47
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../ja_en_data_z5/no_bpe/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4 dicts_rthres 30000" -z 6 -p 1
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4 dicts_rthres 30000" -z 6 -p 3
# --> to 0.2
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../ja_en_data_z5/no_bpe/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000" -z 6 -p 1
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000" -z 6 -p 5

# 17.11.22
# maybe later should change to other corpus, but this one still on wit-en-fr
# (33+, 34+)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.2 idrop_embedding 0.1 drop_hidden 0.2 drop_embedding 0.1 bk_init_enabled ZZ bk_init_nl gaussian bk_init_l gaussian" -p 2
# (35.8, 35.9)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4 bk_init_enabled ZZ no_show_loss" -p 3
# (35.8, 35.6)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4 train_scale_way norm train_scale 1.0" -p 4
# (35.9, 36.0)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4 train_len_uidx 150000 train_len_xadd ZZ train_len_lambda 0.1" -p 5
# (36.0, 36.3)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4 train_len_uidx 100000 train_len_xadd ZZ train_len_xback ZZ train_len_lambda 0.1" -p 6
# (35.9, 35.7)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4 train_len_uidx 100000 train_len_xadd ZZ train_len_xback ZZ train_len_lambda 1.0" -p 7

# 17.11.26
# start to tune the zh_en
# x48
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000" -z 6 -p 3
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.3 drop_hidden 0.3 drop_embedding 0.3 dicts_rthres 30000" -z 6 -p 4
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 40000" -z 6 -p 5
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002" -z 6 -p 6
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "hidden_att 500 gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000" -z 6 -p 7
# x47
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 2
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 3
