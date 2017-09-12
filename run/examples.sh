#!/usr/bin/env bash

GPUID=??
RUNDIR=??
DATADIR=??
SRC=??
TRG=??

set -e

## include xnmt.yaml

# znmt
PYTHONPATH=${DY_ZROOT}/gbuild/python python3.5 ../znmt/train.py --train ${DATADIR}/train.final.{${SRC},${TRG}} --dev ${DATADIR}/dev.final.{${SRC},${TRG}} -v --report_freq 1000 --max_len 50 --max_updates 500000 --no_overwrite --batch_size 64 --valid_batch_size 32 --dynet-devices GPU:${GPUID}

PYTHONPATH=${DY_ZROOT}/gbuild/python python3.5 ../znmt/train.py --train ${DATADIR}/train.final.{${SRC},${TRG}} --dev ${DATADIR}/dev.final.{${SRC},${TRG.restore}} -v --report_freq 1000 --max_len 50 --max_updates 500000 --no_overwrite --batch_size 64 --valid_metric bleu --valid_batch_size 1 --decode_batched --dynet-devices GPU:${GPUID}

PYTHONPATH=${DY_ZROOT}/gbuild/python python3.5 ../znmt/test.py -t ${DATADIR}/test.final.{${SRC},${TRG}.restore} -d ${RUNDIR}/{"src","trg"}.v -m ${RUNDIR}/zbest.model -n 1 --decode_batched --test_batch_size 1 -o output.txt --dynet-devices GPU:${GPUID}

# nematus (need smaller batch-size for 12G-GPU)
CUDA_VISIBLE_DEVICES=${GPUID} PYTHONPATH=../Theano THEANO_FLAGS=FAST_RUN,floatX=float32,device=cuda python2.7 ../nematus/nematus/nmt.py --datasets ${DATADIR}/train.final.{${SRC},${TRG}} --valid_datasets ${DATADIR}/dev.final.{${SRC},${TRG}} --dictionaries ${DATADIR}/vocab.{${SRC},${TRG}}.json --dispFreq 1000 --maxlen 50 --anneal_restarts 2 --finish_after 500000 --batch_size 50 --valid_batch_size 32

PYTHONPATH=../Theano python2.7 ../nematus/nematus/translate.py -i ${DATADIR}/test.final.${SRC} -o output.txt -k 10 -n -m model.npz -v

CUDA_VISIBLE_DEVICES=${GPUID} PYTHONPATH=../Theano THEANO_FLAGS=FAST_RUN,floatX=float32,device=cuda python2.7 ../nematus/nematus/translate.py -i ${DATADIR}/test.final.${SRC} -o output.txt -k 10 -n -m model.npz

ZMT=.. bash ../znmt/scripts/restore.sh <output.txt | perl ../znmt/scripts/multi-bleu.perl -lc ${DATADIR}/test.final.${TRG}.restore

# xnmt
PYTHONPATH=${DY_ZROOT}/gbuild/python:../xnmt/ python3 ../xnmt/xnmt/xnmt_run_experiments.py standard.yaml --dynet-devices GPU:${GPUID}

perl ../znmt/scripts/multi-bleu.perl -lc ${DATADIR}/test.final.${TRG}.restore < run.hyp
