#!/usr/bin/env bash

# znmt
PYTHONPATH=${DY_ZROOT}/gbuild/python python3.5 ../znmt/train.py --train ../data2/en-fr/train.final.{en,fr} --dev ../data2/en-fr/dev.final.{en,fr} --max_len 50 --dynet-devices GPU:??

PYTHONPATH=${DY_ZROOT}/gbuild/python python3.5 ../znmt/train.py --train ../data2/en-fr/train.final.{en,fr} --dev ../data2/en-fr/dev.final.{en,fr.restore} --max_len 50 --valid_metric bleu --dynet-devices GPU:??

MDIR=..
PYTHONPATH=${DY_ZROOT}/gbuild/python python3.5 ../znmt/test.py -t ../data2/en-fr/test.final.{en,fr.restore} -m ${MDIR}/zbest.model -o output.txt -n 1 --decode_batched --test_batch_size 4 -d ${MDIR}/{src.v,trg.v} --dynet-devices GPU:??

# nematus
CUDA_VISIBLE_DEVICES=? PYTHONPATH=../Theano python2.7 ../nematus/nematus/nmt.py --datasets ../data2/en-fr/train.final.{en,fr} --valid_datasets ../data2/en-fr/dev.final.{en,fr} --dictionaries ../data2/en-fr/vocab.{en,fr}.json --maxlen 50

PYTHONPATH=../Theano python2.7 ../nematus/nematus/translate.py -i ../data2/en-fr/test.final.en -o output.txt -k 10 -n -m model.npz

ZMT=.. bash ../znmt/scripts/restore.sh <output.txt | perl ../znmt/scripts/multi-bleu.perl -lc ../data2/en-fr/test.final.fr.restore
