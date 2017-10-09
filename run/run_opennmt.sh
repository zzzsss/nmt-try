#!/usr/bin/env bash

set -e

# pre-process
python3 ${zmt}/OpenNMT-py/preprocess.py -train_src ${datadir}/train.final.${src} -train_tgt ${datadir}/train.final.${trg} -valid_src ${datadir}/dev.final.${src} -valid_tgt ${datadir}/dev.final.${trg} -save_data ./demo

# train (for en-fr on gpu)
python3 ${zmt}/OpenNMT-py/train.py -data ./demo -save_model demo-model -enc_layers 1 -dec_layers 1 -rnn_size 1000 -input_feed 0 -rnn_type GRU -global_attention mlp -gpuid ${gpuid} -seed 12345 -batch_size 80 -optim adam -dropout 0.2 -learning_rate 0.0001 -start_decay_at 40 -epochs 50

# test
python3 ${zmt}/OpenNMT-py/translate.py -model ./demo-model*50.pt -src ${datadir}/test.final.${src} -output ${output} -replace_unk -verbose

#eval
ZMT=${zmt} bash ${zmt}/znmt/scripts/restore.sh <${output} | perl ${zmt}/znmt/scripts/multi-bleu.perl ${datadir}/test.final.${trg}.restore

