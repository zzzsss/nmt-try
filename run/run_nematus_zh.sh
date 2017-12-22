#!/usr/bin/env bash

#set -e

chmod +x ${_nematus_valid_script}

CH_EN_DATADIR=${zmt}/zh_en_data/
DEV_SUBDIR="Dev-set"
TEST_SUBDIR="Test-set"
EVAL_SUBDIR="Reference-for-evaluation"

# nematus (need smaller batch-size for 12G-GPU)
CUDA_VISIBLE_DEVICES=${gpuid} PYTHONPATH=${zmt}/Theano THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=${_nematus_device} python2.7 ${py_args} ${zmt}/nematus/nematus/nmt.py --datasets $CH_EN_DATADIR/train.final.{${src},${trg}} --valid_datasets $CH_EN_DATADIR/$DEV_SUBDIR/nist_2002.{src,ref0} --dictionaries $CH_EN_DATADIR/vocab.{${src},${trg}}.json --dispFreq 1000 --maxlen 50 --finish_after 500000 --batch_size ${batch_size} --valid_batch_size ${valid_batch_width} --external_validation_script ${_nematus_valid_script} --validFreq ${valid_freq} --patience 3 --anneal_restarts 2 ${extras}

# to test
for dataname in nist_2002 nist_2003 nist_2004 nist_2005 nist_2006 nist_2008;
do
time CUDA_VISIBLE_DEVICES=${gpuid} PYTHONPATH=${zmt}/Theano THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=${_nematus_device} python2.7 ${zmt}/nematus/nematus/translate.py -i $CH_EN_DATADIR/$TEST_SUBDIR/$dataname.src -o ${output}.$dataname.n1 -k ${test_beam_size} -n ${normalize} -m ${rundir}/model.npz.dev.npz
perl ${zmt}/znmt/scripts/multi-bleu.perl $CH_EN_DATADIR/$EVAL_SUBDIR/$dataname/$dataname.ref < ${output}.$dataname.n1
done

# timeout for another two days
if echo ${_dy_device} | grep "GPU";
then
timeout 2d python ${zmt}/znmt/run/zhold.py ${zhold} --dynet-devices ${_dy_device} --dynet-mem 4
fi
