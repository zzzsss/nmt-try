#!/usr/bin/env bash

#set -e

chmod +x ${_nematus_valid_script}

# nematus (need smaller batch-size for 12G-GPU)
CUDA_VISIBLE_DEVICES=${gpuid} PYTHONPATH=${zmt}/Theano THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=${_nematus_device} python2.7 ${py_args} ${zmt}/nematus/nematus/nmt.py --datasets ${datadir}/train.final.{${src},${trg}} --valid_datasets ${datadir}/dev.final.{${src},${trg}} --dictionaries ${datadir}/vocab.{${src},${trg}}.json --dispFreq ${report_freq} --maxlen ${max_len} --finish_after ${max_updates} --batch_size ${batch_size} --valid_batch_size ${valid_batch_width} --external_validation_script ${_nematus_valid_script} --validFreq ${valid_freq} --patience ${patience} --anneal_restarts ${anneal_restarts} ${extras}

#PYTHONPATH=${zmt}/Theano python2.7 ${zmt}/nematus/nematus/translate.py -i ${datadir}/test.final.${src} -o output.txt -k 10 -n -m ${rundir}/model.npz -v

time CUDA_VISIBLE_DEVICES=${gpuid} PYTHONPATH=${zmt}/Theano THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=${_nematus_device} python2.7 ${zmt}/nematus/nematus/translate.py -i ${datadir}/test.final.${src} -o ${output} -k ${test_beam_size} -n ${normalize} -m ${rundir}/model.npz.dev.npz

ZMT=${zmt} bash ${zmt}/znmt/scripts/restore.sh <${output} | perl ${zmt}/znmt/scripts/multi-bleu.perl ${datadir}/test.final.${trg}.restore
