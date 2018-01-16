#!/usr/bin/env bash

# final runnings

# en->de
# z1222x47_nematus
#python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t nematus --batch_size 80 --extras "dropout_source 0.2 dropout_target 0.2 n_words_src 50000 n_words 50000 lrate 0.0002 use_dropout" -p 0 --normalize 1.0 --test_beam_size 10
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t nematus --batch_size 64 --valid_batch_width 64 --extras "dropout_source 0.2 dropout_target 0.2 n_words_src 50000 n_words 50000 lrate 0.0002 use_dropout" -p 0 --normalize 1.0 --test_beam_size 10

# zh->en
# z1222x47_nem
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t nematus_zh --batch_size 80 --extras "dropout_source 0.2 dropout_target 0.2 n_words_src 30000 n_words 30000 lrate 0.0002 use_dropout" -p 2 --normalize 1.0 --test_beam_size 10

# --------------------

# ${zmt}/baselines
# x45
#python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0002" -z 8 -p 0
#python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0002 dicts_rthres 30000" -z 8 -p 3

# x45 (baseline-1 -> ed1, ze1)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001" -z 8 -p 0
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts_rthres 30000" -z 8 -p 3
