#!/usr/bin/env bash

#set -e
CH_EN_DATADIR=${zmt}/zh_en_data/
DEV_SUBDIR="Dev-set"

prefix=${rundir}/model.npz
dev=$CH_EN_DATADIR/$DEV_SUBDIR/nist_2002.src
ref=$CH_EN_DATADIR/$DEV_SUBDIR/nist_2002.ref
doutput=dev.final.${trg}

echo "Validating at ..." `date`
MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cpu PYTHONPATH=${zmt}/Theano python2.7 ${zmt}/nematus/nematus/translate.py -i $dev -o $doutput.output.dev -k 1 -n ${normalize} -m $prefix.dev.npz -p 1

## get BLEU
BEST=`cat $prefix.best_bleu || echo 0`
BLEU_REPORT=`perl ${zmt}/znmt/scripts/multi-bleu.perl $ref < $doutput.output.dev`
echo $BLEU_REPORT >> $prefix.bleu_scores
BLEU=`echo $BLEU_REPORT | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "BLEU = $BLEU // $BLEU_REPORT [ for $dev / $ref ]"
# save model with highest BLEU
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > $prefix.best_bleu
  cp $prefix.dev.npz $prefix.npz.best_bleu
fi
