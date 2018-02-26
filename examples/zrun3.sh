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

# TODO: epoch baselines
# x45 (*ev: baseline with epoch-valid)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 no_validate_freq ZZ validate_epoch ZZ max_epochs 10" -z 8 -p 0
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts_rthres 30000 no_validate_freq ZZ validate_epoch ZZ max_epochs 20" -z 8 -p 3
# x44 (ed_raml)
#python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 no_validate_freq ZZ validate_epoch ZZ max_epochs 10 raml_samples 1" -z 8 -p 0
# x47 (ze_raml)
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 no_validate_freq ZZ validate_epoch ZZ max_epochs 20 raml_samples 1" -z 8 -p 3

# other baselines
# x47: # ${zmt}/baselines2/ze_ev
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.00005 dicts ../../baselines/ze_ev/{src,trg}.v reload_model_name ../../baselines/ze_ev/zbest.model reload ZZ no_reload_training_progress ZZ no_rebuild_dicts ZZ valid_freq 5000 validate0 ZZ" -p 0
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.00005 dicts ../../baselines/ze_ev/{src,trg}.v reload_model_name ../../baselines/ze_ev/zbest.model reload ZZ no_reload_training_progress ZZ no_rebuild_dicts ZZ valid_freq 5000 validate0 ZZ raml_samples 1" -p 1
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.00005 dicts ../../baselines/ze_ev/{src,trg}.v reload_model_name ../../baselines/ze_ev/zbest.model reload ZZ no_reload_training_progress ZZ no_rebuild_dicts ZZ valid_freq 5000 validate0 ZZ ss_mode isigm ss_scale 5000 ss_k 10" -p 2
#
#python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../../baselines/ed1/{src,trg}.v reload_model_name ../../baselines/ed1/zbest.model reload ZZ no_reload_training_progress ZZ no_rebuild_dicts ZZ valid_freq 5000 validate0 ZZ" -p -1
#python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../../baselines/ed1/{src,trg}.v reload_model_name ../../baselines/ed1/zbest.model reload ZZ no_reload_training_progress ZZ no_rebuild_dicts ZZ valid_freq 5000 validate0 ZZ raml_samples 1" -p -1
#python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../../baselines/ed1/{src,trg}.v reload_model_name ../../baselines/ed1/zbest.model reload ZZ no_reload_training_progress ZZ no_rebuild_dicts ZZ valid_freq 5000 validate0 ZZ ss_mode linear ss_scale 10000 ss_k 0.1" -p -1

# no drops baseline --once-shuffle
# x44/x45, ze_ev_nodrop/ze_ev_drop-1-2-3
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.0 idrop_embedding 0.0 drop_hidden 0.0 drop_embedding 0.0 lrate 0.0001 dicts_rthres 30000 no_validate_freq ZZ validate_epoch ZZ max_epochs 50 shuffle_training_data_onceatstart ZZ no_shuffle_training_data ZZ patience 10 anneal_restarts 0" -z 8 -p 0
# -> ze_ev_drop2: patience=5
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts_rthres 30000 no_validate_freq ZZ validate_epoch ZZ max_epochs 50 shuffle_training_data_onceatstart ZZ no_shuffle_training_data ZZ patience 10 anneal_restarts 0" -z 8 -p 0
#
#
# -> ze_ev_drop3: p=10, max_up=1000000
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts_rthres 30000 no_validate_freq ZZ validate_epoch ZZ max_epochs 50 shuffle_training_data_onceatstart ZZ no_shuffle_training_data ZZ patience 10 anneal_restarts 0 max_updates 1000000" -z 4 -p 0
# -> ze_ev_drop4: p=10, max_up=1000000, max_epoch=100
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts_rthres 30000 no_validate_freq ZZ validate_epoch ZZ max_epochs 100 shuffle_training_data_onceatstart ZZ no_shuffle_training_data ZZ patience 10 anneal_restarts 0 max_updates 1000000" -z 4 -p 0

# ed_ev_drop
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 no_validate_freq ZZ validate_epoch ZZ max_epochs 25 shuffle_training_data_onceatstart ZZ no_shuffle_training_data ZZ patience 5 anneal_restarts 0 max_updates 10000000" -z 4 -p 0
# no-bpe of ed
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/no-bpe/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts_rthres 30000 no_validate_freq ZZ validate_epoch ZZ max_epochs 25 shuffle_training_data_onceatstart ZZ no_shuffle_training_data ZZ patience 5 anneal_restarts 0 max_updates 10000000" -z 4 -p 0

# ramlx47
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts_rthres 30000 no_validate_freq ZZ validate_epoch ZZ max_epochs 50 shuffle_training_data_onceatstart ZZ no_shuffle_training_data ZZ patience 10 anneal_restarts 0 max_updates 1000000 raml_samples 1" -z 4 -p 0
