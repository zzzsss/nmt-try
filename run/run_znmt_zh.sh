#!/usr/bin/env bash

# special running for chinese ones
# set -e

if echo ${_dy_device} | grep "GPU";
then
export PYTHONPATH=$DY_ZROOT/gbuild/python;
else
export PYTHONPATH=$DY_ZROOT/cbuild/python;
fi

CH_EN_DATADIR=${zmt}/zh_en_data/
DEV_SUBDIR="Dev-set"
TEST_SUBDIR="Test-set"
EVAL_SUBDIR="Reference-for-evaluation"

python3.5 ${py_args} ${zmt}/znmt/train.py -v --no_overwrite --eval_metric ibleu --train $CH_EN_DATADIR/train.final.{${src},${trg}} --dev $CH_EN_DATADIR/$DEV_SUBDIR/nist_2002.{src,ref0} --dynet-devices ${_dy_device} ${extras}
if [ -r stat.prof ]; then mv stat.prof _train.prof; fi

# to test
for dataname in nist_2002 nist_2003 nist_2004 nist_2005 nist_2006 nist_2008;
do
python3.5 ${py_args} ${zmt}/znmt/test.py -v --report_freq 125 --eval_metric ibleu -t $CH_EN_DATADIR/$TEST_SUBDIR/$dataname.src $CH_EN_DATADIR/$EVAL_SUBDIR/$dataname/$dataname.ref0 -d ${rundir}/{"src","trg"}.v -m ${rundir}/zbest.model -n 1.0 -o ${output}.$dataname.n1 --normalize_way norm --dynet-devices ${_dy_device} --beam_size ${test_beam_size} ${extras}
perl ${zmt}/znmt/scripts/multi-bleu.perl $CH_EN_DATADIR/$EVAL_SUBDIR/$dataname/$dataname.ref < ${output}.$dataname.n1
done

# timeout for another two days
if echo ${_dy_device} | grep "GPU";
then
timeout 2d python3.5 ${zmt}/znmt/run/zhold.py ${zhold} --dynet-devices ${_dy_device} --dynet-mem 4
fi
