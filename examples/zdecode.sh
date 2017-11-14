#!/usr/bin/env bash

zmt="../.."
rundir="."
datadir="../../data2/wit3-en-fr_z5/"
output="zdecode.txt"
src="en"
trg="fr"

PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 -m pdb ${zmt}/znmt/test.py -v --report_freq 125 --beam_size 5 -o ${output} -t ${datadir}/test.final.{${src},${trg}.restore} -d ${rundir}/{"src","trg"}.v -m ${rundir}/zbest.model --dynet-devices GPU:0
ZMT=${zmt} bash ${zmt}/znmt/scripts/restore.sh <${output} | perl ${zmt}/znmt/scripts/multi-bleu.perl ${datadir}/test.final.${trg}.restore

