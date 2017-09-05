#!/usr/bin/env bash

PYTHONPATH=${DY_ZROOT}/gbuild/python python3.5 ../znmt/train.py --train ../data2/en-fr/train.final.{en,fr} --dev ../data2/en-fr/dev.final.{en,fr} --dynet-devices ??

PYTHONPATH=${DY_ZROOT}/gbuild/python python3.5 ../znmt/train.py --train ../data2/en-fr/train.final.{en,fr} --dev ../data2/en-fr/dev.final.{en,fr.restore} --valid_metric bleu --dynet-devices ??

PYTHONPATH=${DY_ZROOT}/gbuild/python python3.5 ../znmt/train.py --test ../data2/en-fr/test.final.{en,fr.restore}

