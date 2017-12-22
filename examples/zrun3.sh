#!/usr/bin/env bash

# final runnings

# en->de
# z1222x47_nematus
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t nematus --batch_size 80 --extras "dropout_source 0.2 dropout_target 0.2 n_words_src 50000 n_words 50000 lrate 0.0002 use_dropout" -p 0 --normalize 1.0 --test_beam_size 10

# zh->en

