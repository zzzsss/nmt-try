#!/usr/bin/env bash

#set -e

# xnmt (include _xnmt.yaml)
PYTHONPATH=$DY_ZROOT/gbuild/python:${zmt}/xnmt/ python3 ${py_args} ${zmt}/xnmt/xnmt/xnmt_run_experiments.py ${_xnmt_yaml} --dynet-devices GPU:${gpuid} --dynet-seed 12345

# todo: not available for xnmt yet
# PYTHONPATH=$DY_ZROOT/gbuild/python:${zmt}/xnmt/ python3 ${zmt}/xnmt/xnmt/xnmt_decode.py --beam 10 --post_process join-bpe --dynet-devices GPU:${gpuid} --model_file ${rundir}/run.mod --len_norm_type PolynomialNormalization ${datadir}/test.final.${src} ${output} --dynet-seed 12345
cp ./run.out ${output}

ZMT=${zmt} bash ${zmt}/znmt/scripts/restore.sh <${output} | perl ${zmt}/znmt/scripts/multi-bleu.perl ${datadir}/test.final.${trg}.restore
