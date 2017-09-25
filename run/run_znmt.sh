#!/usr/bin/env bash

#set -e

PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ${zmt}/znmt/train.py -v --no_overwrite --train ${datadir}/train.final.{${src},${trg}} --dev ${datadir}/dev.final.{${src},${trg}} --max_updates ${max_updates} --max_len ${max_len} --batch_size ${batch_size} --valid_batch_width ${valid_batch_width} --report_freq ${report_freq} --beam_size ${dev_beam_size} -n ${normalize} --valid_freq ${valid_freq} --patience ${patience} --anneal_restarts ${anneal_restarts} --dynet-devices GPU:${gpuid} --dynet-seed 12345 ${extras}

PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ${zmt}/znmt/test.py -v -t ${datadir}/test.final.{${src},${trg}.restore} -d ${rundir}/{"src","trg"}.v -m ${rundir}/zbest.model -n ${normalize} -o ${output} --dynet-devices GPU:${gpuid} --beam_size ${test_beam_size} --dynet-seed 12345 ${extras}

ZMT=${zmt} bash ${zmt}/znmt/scripts/restore.sh <${output} | perl ${zmt}/znmt/scripts/multi-bleu.perl ${datadir}/test.final.${trg}.restore
