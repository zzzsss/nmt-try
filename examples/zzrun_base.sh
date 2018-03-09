#!/usr/bin/env bash

# rerun with more comparable params

# 1. basic runs from the start (x47)
# basic ze (len-p 1.0)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts_rthres 30000 no_validate_freq ZZ validate_epoch ZZ shuffle_training_data_onceatstart ZZ no_shuffle_training_data ZZ patience 10 normalize_way add normalize_alpha 1.0" -z 4 -p 2

# raml
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts_rthres 30000 no_validate_freq ZZ validate_epoch ZZ shuffle_training_data_onceatstart ZZ no_shuffle_training_data ZZ patience 10 normalize_way add normalize_alpha 1.0 raml_samples 1" -z 4 -p 3

# basic ed (no len-p)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 no_validate_freq ZZ validate_epoch ZZ shuffle_training_data_onceatstart ZZ no_shuffle_training_data ZZ patience 5 normalize_way add normalize_alpha 0.0" -z 4 -p 4

# raml (##slightly shorter max-len because of 12g-memory restriction)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 no_validate_freq ZZ validate_epoch ZZ shuffle_training_data_onceatstart ZZ no_shuffle_training_data ZZ patience 5 normalize_way add normalize_alpha 0.0 raml_samples 1" -z 4 -p 5

