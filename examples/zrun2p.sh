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
# base: continue learning
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 valid_freq 1000" -z 6 -p 1
# -- based on first restart point: 33.64 for greedy
# seems not good if too diverged from gold -> laso
# -> always only 26+ with norm, trying no norming ...
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 valid_freq 1000 t2_compare_at none" -z 6 -p 0
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 valid_freq 1000 t2_gi_mode laso t2_compare_at none" -z 6 -p 1
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 2 valid_freq 1000 t2_gi_mode laso t2_impl_bsize 30 t2_compare_at none" -z 6 -p 2
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 3 valid_freq 1000 t2_gi_mode laso t2_impl_bsize 20 t2_compare_at none" -z 6 -p 3
#
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 t2_sync_nga ZZ t2_nga_range 5 valid_freq 1000 t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 1 t2_err_gold_lambda 1.0" -z 6 -p 5
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 t2_sync_med ZZ valid_freq 1000 t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 1 t2_err_gold_lambda 1.0" -z 6 -p 7

# todo(question): does dropout matter?

# x48
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../data2/de-en_2014/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z0104x48_b_0/zbest.model no_rebuild_dicts ZZ dicts ../z0104x48_b_0/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 valid_freq 1000 t2_gi_mode laso t2_compare_at none validate0" -z 6 -p 0
#
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../data2/de-en_2014/ -t znmt --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z0104x48_b_2/zbest.model no_rebuild_dicts ZZ dicts ../z0104x48_b_2/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 valid_freq 1000 t2_gi_mode laso t2_compare_at none validate0" -z 6 -p 3

# x47
# -> but will drop to 35- at anneal point
# RR: 36.94 start, -> 37.87
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 t2_sync_nga ZZ t2_nga_n 3 t2_nga_range 5 valid_freq 5000 t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 2 t2_err_gold_mode gold validate0" -z 6 -p 1
# RR: 38.23 at u30000
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 t2_sync_nga ZZ t2_nga_n 4 t2_nga_range 5 valid_freq 5000 t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 2 t2_err_gold_mode gold validate0" -z 6 -p 2
# RR: 37.8 at u35000
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 t2_sync_nga ZZ t2_nga_n 5 t2_nga_range 5 valid_freq 5000 t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 2 t2_err_gold_mode gold validate0" -z 6 -p 3
# -> but these ones will not drop that much at anneal points
# RR: best 37.5 at u70000
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 t2_sync_med ZZ valid_freq 5000 t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 1 t2_err_gold_mode gold validate0" -z 6 -p 5
# RR: best 38.0 at u60000
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 dicts_rthres 30000 lrate 0.0001 reload ZZ no_reload_training_progress ZZ reload_model_name ../z1217_base/zr1.model no_rebuild_dicts ZZ dicts ../z1217_base/{src,trg}.v report_freq 100 train_mode beam train_margin 2.0 t2_beam_size 1 t2_sync_med ZZ valid_freq 5000 t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 2 t2_err_gold_mode gold validate0" -z 6 -p 7

# ===================================
# debug at 180118
# (option) -- batch_size 64 t2_impl_bsize 32 validate0 ZZ
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../../baselines/ze/{src,trg}.v reload_model_name ../../baselines/ze/zp2.m reload ZZ no_reload_training_progress ZZ no_rebuild_dicts ZZ valid_freq 5000 report_freq 100 train_mode beam t2_beam_size 1 t2_sync_med ZZ t2_beam_loss err t2_bad_lambda 0.1 t2_bad_maxlen 0 t2_err_gold_mode gold t2_err_seg_minlen 3 t2_err_mcov_thresh 0.25 t2_err_pright_thresh 0.85 validate_epoch ZZ no_validate_freq ZZ rand_skip 0.5" -z 6 -p 0
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../z1217_base/{src,trg}.v reload_model_name ../z1217_base/zr1.model reload ZZ no_reload_training_progress ZZ no_rebuild_dicts ZZ valid_freq 5000 validate0 ZZ train_mode beam t2_beam_size 1 t2_sync_med ZZ t2_beam_loss err t2_bad_maxlen 0 t2_err_gold_mode based t2_err_seg_minlen 4 t2_err_match_addfirst ZZ report_freq 200" -z 6 -p 0
# -- debug print
python3 ../../znmt/run/zprepare.py --debug --zmt ../.. -d ../../zh_en_data/ -t znmt_zh --extras "gdrop_rec 0.2 idrop_embedding 0.0 drop_hidden 0.2 drop_embedding 0.2 lrate 0.0001 dicts ../../baselines/ze1/{src,trg}.v reload_model_name ../../baselines/ze1/zbest.model reload ZZ no_reload_training_progress ZZ no_rebuild_dicts ZZ valid_freq 5000 train_mode beam t2_beam_size 1 t2_sync_med ZZ t2_beam_loss err t2_bad_maxlen 0 t2_err_gold_mode based t2_err_seg_minlen 4 t2_err_debug_print ZZ max_epochs 1" -z 6 -p 0
