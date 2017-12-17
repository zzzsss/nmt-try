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
# killed
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../ja_en_data_z5/no_bpe/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "dim_word 500 hidden_att 500 rnn_type gru2 gdrop_rec 0.4 idrop_embedding 0.4 drop_hidden 0.4 drop_embedding 0.4 dicts_rthres 30000" -z 6 -p 1
# killed
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
# (35.99/38.58/34.08/32.90/26.11)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000" -z 6 -p 3
# (35.65/38.96/34.68/33.15/26.38)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.3 drop_hidden 0.3 drop_embedding 0.3 dicts_rthres 30000" -z 6 -p 4
# (36.37/39.00/34.31/33.43/26.96)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 40000" -z 6 -p 5
# (36.94/39.61/35.52/34.22/27.38)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002" -z 6 -p 6
# (36.62/39.02/34.73/33.29/26.79)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "hidden_att 500 gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000" -z 6 -p 7
# x47
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 2
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 3

# x46 (1202)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.1 drop_hidden 0.2 drop_embedding 0.1" -z 6 -p 7

# ---------------- (not finished or killed until ?)
# x45 (1202) (dev 19.67)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0002" -z 10 -p 0
# x45 (1208)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 128 --valid_batch_width 128 --extras "gdrop_rec 0.2 idrop_embedding 0.1 drop_hidden 0.2 drop_embedding 0.1 lrate 0.0002" -z 6 -p 0

# 17.12.04/05: final tuning on zh_en and en_de
# x48:
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001" -z 6 -p 4
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.1 drop_hidden 0.2 drop_embedding 0.1 dicts_rthres 30000 lrate 0.0001" -z 6 -p 5
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002" -z 6 -p 6
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.1 drop_hidden 0.2 drop_embedding 0.1 dicts_rthres 30000 lrate 0.0002" -z 6 -p 7
# x47 (ch-r2l)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002" -z 6 -p 7
# x47:
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001" -z 6 -p 0
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.1 drop_hidden 0.2 drop_embedding 0.1 lrate 0.0001" -z 6 -p 1
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0002" -z 6 -p 2
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.1 drop_hidden 0.2 drop_embedding 0.1 lrate 0.0002" -z 6 -p 3

# 17.12.13: baseline and others
# x45 (rerun en-de again)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0002" -z 10 -p 0
# x48
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002" -z 6 -p 3
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002 dec_type ngram" -z 6 -p 4
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002 dec_type ngram" -z 6 -p 5
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002 dec_type ngram dec_depth 2" -z 6 -p 6
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002 dec_type ngram dec_depth 2" -z 6 -p 1
# x47
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002 train_margin 1.0" -z 6 -p 0
# ============= change these ones
# -- (scores too small and could not distinguish)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002 train_margin 1.0 train_local_loss hinge_max" -z 6 -p 1
# -> seems reasonable
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002 train_margin 1.0 train_local_loss hinge_avg batch_size 64" -z 6 -p 2
# -- (maybe lrate too much and too large score-diff; uncontrolled loss)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002 train_margin 1.0 train_local_loss hinge_sum" -z 6 -p 3
#
#
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.1 trainer_type sgd train_margin 2.0" -z 6 -p 1
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.1 trainer_type sgd train_margin 2.0 train_local_loss hinge_max" -z 6 -p 2
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.1 trainer_type sgd train_margin 2.0 train_local_loss hinge_avg batch_size 64" -z 6 -p 3

# not good with n-gram or others(re-run)

# 17.12.17
# after the bug, should set idrop_embedding to 0.
# x45
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0002" -z 10 -p 0
# x46
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002" -z 6 -p 7
# x47
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 1.0 trainer_type sgd train_margin 2.0" -z 6 -p 0
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 1.0 trainer_type sgd train_margin 2.0 train_local_loss hinge_max" -z 6 -p 2
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 1.0 trainer_type sgd train_margin 2.0 train_local_loss hinge_avg batch_size 64" -z 6 -p 3
# x48
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002 dec_type ngram hidden_dec 500 dec_ngram_n 6" -z 6 -p 3
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002 dec_type ngram hidden_dec 500 dec_ngram_n 8" -z 6 -p 4
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.4 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002 dec_type ngram hidden_dec 500 dec_ngram_n 10" -z 6 -p 5
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002 dec_type ngram hidden_dec 500 dec_ngram_n 12" -z 6 -p 6
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --batch_size 80 --valid_batch_width 80 --extras "gdrop_rec 0.4 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0002 dec_type ngram hidden_dec 500 dec_ngram_n 14" -z 6 -p 7
