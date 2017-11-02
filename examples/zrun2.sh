#!/usr/bin/env bash

# 17.11.07 rerun the new ones
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --test_beam_size 1 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2" -z 6 -p 5
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --test_beam_size 1 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 train_scale 1.0" -z 6 -p 6
python3 ../../znmt/run/zprepare.py --profile --zmt ../.. -d ../../data2/wit3-en-fr_z5/ -t znmt --batch_size 80 --patience 3 --test_beam_size 1 --extras "summ_type ends gdrop_rec 0.4 idrop_embedding 0.2 drop_hidden 0.2 drop_embedding 0.2 train_scale 1.0 lrate 0.0002" -z 6 -p 7
