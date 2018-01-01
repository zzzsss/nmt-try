#!/usr/bin/env bash

# beam mode
#python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zbest.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v" -z 6 -p 7

# debug0
# python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../data2/de-en_2014/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001" -z 6 -p 7

# t1: no-gi
# -> sync methods
# (d1)
# report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 2 valid_freq 1000
# (d2)
# report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 2 t2_sync_nga ZZ t2_nga_range 5 valid_freq 1000
# (d3)
# report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 2 t2_sync_med ZZ valid_freq 1000
# -> loss
# t2_beam_loss per
# (d4: d2+err, d5: d3+err)
# t2_beam_loss err
# -> gi, beam_up
# t2_gi_mode none
# t2_gi_mode laso
# t2_gi_mode ngab
# -> bad_lambda, compare_at, t2_err_gold_lambda

# x47
#python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 valid_freq 1000" -z 6 -p 1
#python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 t2_sync_nga ZZ t2_nga_range 5 valid_freq 1000" -z 6 -p 2
#python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 t2_sync_med ZZ valid_freq 1000" -z 6 -p 3
# (only no-drop(slightly increase) one: bad_maxlen=2, ratio=1.5, load=zbest)
# -- this one seems potential
#python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 t2_sync_nga ZZ t2_nga_range 5 valid_freq 1000 t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 1" -z 6 -p 5
# -- not that bad, but still drops a little
#python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 t2_sync_med ZZ valid_freq 1000 t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 1" -z 6 -p 7

# x47
# -- based on first restart point: 33.64 for greedy
# seems not good if too diverged from gold -> laso
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 valid_freq 1000 t2_gi_mode laso" -z 6 -p 1
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 2 valid_freq 1000 t2_gi_mode laso t2_impl_bsize 30" -z 6 -p 2
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 3 valid_freq 1000 t2_gi_mode laso t2_impl_bsize 20" -z 6 -p 3
#
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 t2_sync_nga ZZ t2_nga_range 5 valid_freq 1000 t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 1 t2_err_gold_lambda 1.0" -z 6 -p 5
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 t2_sync_med ZZ valid_freq 1000 t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 1 t2_err_gold_lambda 1.0" -z 6 -p 7
