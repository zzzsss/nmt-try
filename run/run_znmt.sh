#!/usr/bin/env bash

#set -e

if echo ${_dy_device} | grep "GPU";
then
export PYTHONPATH=$DY_ZROOT/gbuild/python;
else
export PYTHONPATH=$DY_ZROOT/cbuild/python;
fi

# should be using python3.5
#PY="python"
#PY="python3.5"

python3.5 ${py_args} ${zmt}/znmt/train.py -v --no_overwrite --train ${datadir}/train.final.{${src},${trg}} --dev ${datadir}/dev.final.{${src},${trg}} --max_updates ${max_updates} --max_len ${max_len} --batch_size ${batch_size} --valid_batch_width ${valid_batch_width} --report_freq ${report_freq} --beam_size ${dev_beam_size} -n ${normalize} --valid_freq ${valid_freq} --patience ${patience} --anneal_restarts ${anneal_restarts} --dynet-devices ${_dy_device} --dynet-seed 12345 ${extras}
if [ -r stat.prof ]; then mv stat.prof _train.prof; fi

python3.5 ${py_args} ${zmt}/znmt/test.py -v --report_freq 125 -t ${datadir}/test.final.{${src},${trg}.restore} -d ${rundir}/{"src","trg"}.v -m ${rundir}/zbest.model -n ${normalize} -o ${output} --dynet-devices ${_dy_device} --beam_size ${test_beam_size} --dynet-seed 12345 ${extras}
ZMT=${zmt} bash ${zmt}/znmt/scripts/restore.sh <${output} | perl ${zmt}/znmt/scripts/multi-bleu.perl ${datadir}/test.final.${trg}.restore
if [ -r stat.prof ]; then mv stat.prof _test.prof; fi

# some extras with specific normalizations
python3.5 ${py_args} ${zmt}/znmt/test.py -v --report_freq 125 -t ${datadir}/test.final.{${src},${trg}.restore} -d ${rundir}/{"src","trg"}.v -m ${rundir}/zbest.model -n 0.0 -o ${output} --dynet-devices ${_dy_device} --beam_size ${test_beam_size} --dynet-seed 12345 ${extras}
ZMT=${zmt} bash ${zmt}/znmt/scripts/restore.sh <${output} | perl ${zmt}/znmt/scripts/multi-bleu.perl ${datadir}/test.final.${trg}.restore

python3.5 ${py_args} ${zmt}/znmt/test.py -v --report_freq 125 -t ${datadir}/test.final.{${src},${trg}.restore} -d ${rundir}/{"src","trg"}.v -m ${rundir}/zbest.model -n 1.0 -o ${output} --dynet-devices ${_dy_device} --beam_size ${test_beam_size} --dynet-seed 12345 ${extras}
ZMT=${zmt} bash ${zmt}/znmt/scripts/restore.sh <${output} | perl ${zmt}/znmt/scripts/multi-bleu.perl ${datadir}/test.final.${trg}.restore

python3.5 ${py_args} ${zmt}/znmt/test.py -v --report_freq 125 -t ${datadir}/test.final.{${src},${trg}.restore} -d ${rundir}/{"src","trg"}.v -m ${rundir}/zbest.model -n 1.0 --normalize_way google -o ${output} --dynet-devices ${_dy_device} --beam_size ${test_beam_size} --dynet-seed 12345 ${extras}
ZMT=${zmt} bash ${zmt}/znmt/scripts/restore.sh <${output} | perl ${zmt}/znmt/scripts/multi-bleu.perl ${datadir}/test.final.${trg}.restore

# timeout for another two days
if echo ${_dy_device} | grep "GPU";
then
timeout 2d python3.5 ${zmt}/znmt/run/zhold.py ${zhold} --dynet-devices ${_dy_device} --dynet-mem 4
fi
