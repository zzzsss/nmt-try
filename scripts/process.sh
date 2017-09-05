#!/usr/bin/env bash

# process the data with several steps

# special ARGS and VARS: $1=script, $BPE_OP=num-bpe(has its default)

set -e
set -v

RUNNING_DIR="$( cd "$( dirname ${BASH_SOURCE[0]}  )" && pwd )"
INCLU=$1
if [ ! -r $INCLU ]
then
INCLU=${RUNNING_DIR}/$1
elif [ ! -r $INCLU ]
then
INCLU=${RUNNING_DIR}/$1.sh
else
echo "Error, haven't find shell to run: $1"
fi

source $INCLU
shopt -s expand_aliases

time pstep1-train

time pstep2-dt

time pstep3-process
POSTFIX_TR="tok.clean.tc"
POSTFIX_DT="tok.tc"

if [ -z "$BPE_OP" ]; then BPE_OP=50000; fi
if [ -z "$BPE_CUT" ]; then BPE_CUT=50000; fi
if [ "$BPE_OP" == "0" ]
then echo "Skip bpe!!"
else
time pstep4-bpe $BPE_OP $BPE_CUT
POSTFIX_TR="tok.clean.tc.bpe"
POSTFIX_DT="tok.tc.bpe"
fi

time pstep5-concat
