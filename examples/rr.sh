#!/usr/bin/env bash

# usage: bash rr.sh dir_prefix start_id

#set -v
shopt -s expand_aliases

echo "# dir_prefix:$1, start_id:$2"
echo "# step1: kill these zhold programs"
KILL_PROG="zhold.py"
ps -elf | grep ${KILL_PROG} | while read line; do
echo "# KILL: ${line}"
done
echo "# step2: build dir and run new programs, read from input"
NEW_PROGRAM="python3"
CUR_DIR=`pwd`
CUR_NUM=$2
cat | grep ${NEW_PROGRAM} | while read line; do
if [ ${CUR_NUM} -eq $2 ]; then
    echo; echo; echo "# START commands !!"; echo "zpkill ${KILL_PROG}"
fi
if [[ ${line} =~ "^#.*" ]]; then
    echo ${line}
else
    NEW_DIR="${CUR_DIR}/$1_${CUR_NUM}"
    echo "mkdir ${NEW_DIR}; cd ${NEW_DIR}"
    echo "${line}"
    echo "bash -v _test.sh >log 2>&1 &"
    CUR_NUM=$((CUR_NUM + 1))
fi
done
echo "cd ${CUR_DIR}"
