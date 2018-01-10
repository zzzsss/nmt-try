#!/usr/bin/env bash

# start on 180114, real the final ones

# =======================
# TODO(FINAL2): again, only the basic baselines ...
# {zmt}/baselines
# x45
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0002" -z 8 -p 0
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0002 dicts_rthres 30000" -z 8 -p 3

# x47 on en_de
# continue
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../z1217x45_base/{src,trg}.v reload_model_name ../z1217x45_base/zbest.model reload ZZ no_reload_training_progress ZZ  no_rebuild_dicts ZZ valid_freq 2500 validate0 ZZ" -z 6 -p 0
# raml
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../z1217x45_base/{src,trg}.v reload_model_name ../z1217x45_base/zbest.model reload ZZ no_reload_training_progress ZZ  no_rebuild_dicts ZZ valid_freq 2500 raml_samples 1 validate0 ZZ" -z 6 -p 1
# err
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../z1217x45_base/{src,trg}.v reload_model_name ../z1217x45_base/zbest.model reload ZZ no_reload_training_progress ZZ  no_rebuild_dicts ZZ valid_freq 2500 validate0 ZZ train_mode beam t2_beam_size 1 t2_sync_med ZZ t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 0 t2_err_gold_mode gold" -z 6 -p 2
# err
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../z1217x45_base/{src,trg}.v reload_model_name ../z1217x45_base/zbest.model reload ZZ no_reload_training_progress ZZ  no_rebuild_dicts ZZ valid_freq 2500 validate0 ZZ train_mode beam t2_beam_size 1 t2_sync_med ZZ t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 1 t2_err_gold_mode gold" -z 6 -p 3
# err
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../z1217x45_base/{src,trg}.v reload_model_name ../z1217x45_base/zbest.model reload ZZ no_reload_training_progress ZZ  no_rebuild_dicts ZZ valid_freq 2500 validate0 ZZ train_mode beam t2_beam_size 1 t2_sync_med ZZ t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 2 t2_err_gold_mode gold" -z 6 -p 5
# err
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../en_de_data_z5/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../z1217x45_base/{src,trg}.v reload_model_name ../z1217x45_base/zbest.model reload ZZ no_reload_training_progress ZZ  no_rebuild_dicts ZZ valid_freq 2500 validate0 ZZ train_mode beam t2_beam_size 1 t2_sync_med ZZ t2_beam_loss err t2_bad_lambda 1.0 t2_bad_maxlen 2 t2_err_gold_mode gold" -z 6 -p 7

# x48 on zh_en
# continue
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../z1217_base/{src,trg}.v reload_model_name ../z1217_base/zbest.model reload ZZ no_reload_training_progress ZZ no_rebuild_dicts ZZ valid_freq 2500 validate0 ZZ" -z 6 -p 0
# raml
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../z1217_base/{src,trg}.v reload_model_name ../z1217_base/zbest.model reload ZZ no_reload_training_progress ZZ no_rebuild_dicts ZZ valid_freq 2500 raml_samples 1 validate0 ZZ" -z 6 -p 3
# --> RR: but they all drop about 0.5 ...
# err
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../z1217_base/{src,trg}.v reload_model_name ../z1217_base/zbest.model reload ZZ no_reload_training_progress ZZ no_rebuild_dicts ZZ valid_freq 2500 validate0 ZZ train_mode beam t2_beam_size 1 t2_sync_med ZZ t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 0 t2_err_gold_mode gold" -z 6 -p 5
# err
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../z1217_base/{src,trg}.v reload_model_name ../z1217_base/zbest.model reload ZZ no_reload_training_progress ZZ no_rebuild_dicts ZZ valid_freq 2500 validate0 ZZ train_mode beam t2_beam_size 1 t2_sync_med ZZ t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 1 t2_err_gold_mode gold" -z 6 -p 6
# err
python3 ../../znmt/run/zprepare.py --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../z1217_base/{src,trg}.v reload_model_name ../z1217_base/zbest.model reload ZZ no_reload_training_progress ZZ no_rebuild_dicts ZZ valid_freq 2500 validate0 ZZ train_mode beam t2_beam_size 1 t2_sync_med ZZ t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 2 t2_err_gold_mode gold" -z 6 -p 7
