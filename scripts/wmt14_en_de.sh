#!/usr/bin/env bash

# DE && EN
# this file is from seq2seq: https://github.com/google/seq2seq/blob/master/bin/data/wmt16_en_de.sh
# also refer to https://github.com/rsennrich/wmt16-scripts
# -- running from pwd, treat it as home dir

set -e
set -v

RUNNING_DIR="$( cd "$( dirname ${BASH_SOURCE[0]}  )" && pwd )"
source ${RUNNING_DIR}/basic.sh
shopt -s expand_aliases

SRC="en"
TRG="de"

CUR_DATA_DIR="${HOME_DIR}/${SRC}_${TRG}_data"
mkdir -p ${CUR_DATA_DIR}

# Concatenate training data (lines: 4590101)
cat "${DATA_DIR}/europarl-v7/training/europarl-v7.de-en.en" \
  "${DATA_DIR}/commoncrawl/commoncrawl.de-en.en" \
  "${DATA_DIR}/nc-v12/training/news-commentary-v12.de-en.en" \
  > "${CUR_DATA_DIR}/train.en"
wc "${CUR_DATA_DIR}/train.en"
cat "${DATA_DIR}/europarl-v7/training/europarl-v7.de-en.de" \
  "${DATA_DIR}/commoncrawl/commoncrawl.de-en.de" \
  "${DATA_DIR}/nc-v12/training/news-commentary-v12.de-en.de" \
  > "${CUR_DATA_DIR}/train.de"
wc "${CUR_DATA_DIR}/train.de"

# get dev and test (start with news*)
cp ${DATA_DIR}/dev/dev/newstest2012.{${SRC},${TRG}} ${CUR_DATA_DIR}
cp ${DATA_DIR}/dev/dev/newstest2013.{${SRC},${TRG}} ${CUR_DATA_DIR}
input-from-sgm <"${DATA_DIR}/dev/dev/newstest2014-deen-src.${SRC}.sgm" >"${CUR_DATA_DIR}/newstest2014.${SRC}"
input-from-sgm <"${DATA_DIR}/dev/dev/newstest2014-deen-ref.${TRG}.sgm" >"${CUR_DATA_DIR}/newstest2014.${TRG}"
input-from-sgm <"${DATA_DIR}/dev/dev/newstest2015-${SRC}${TRG}-src.${SRC}.sgm" >"${CUR_DATA_DIR}/newstest2015.${SRC}"
input-from-sgm <"${DATA_DIR}/dev/dev/newstest2015-${SRC}${TRG}-ref.${TRG}.sgm" >"${CUR_DATA_DIR}/newstest2015.${TRG}"
input-from-sgm <"${DATA_DIR}/dev/dev/newstest2016-${SRC}${TRG}-src.${SRC}.sgm" >"${CUR_DATA_DIR}/newstest2016.${SRC}"
input-from-sgm <"${DATA_DIR}/dev/dev/newstest2016-${SRC}${TRG}-ref.${TRG}.sgm" >"${CUR_DATA_DIR}/newstest2016.${TRG}"
for ff in ${CUR_DATA_DIR}/news*; do
    fname=`basename ${ff}`
    mv ${ff} ${CUR_DATA_DIR}/dt.${fname}   # dev or test
done

time prepare-data "${CUR_DATA_DIR}" "${SRC}" "${TRG}"
time bpe-join "${CUR_DATA_DIR}" "${SRC}" "${TRG}" 90000

########### -- task specific -- ###########
for lang in ${SRC} ${TRG}; do
    rm ${CUR_DATA_DIR}/train.final.${lang}
    ln -s ${CUR_DATA_DIR}/train.tok.clean.tc.bpe.${lang} ${CUR_DATA_DIR}/train.final.${lang}
    cat ${CUR_DATA_DIR}/dt.newstest201{2,3}.tok.tc.bpe.${lang} > ${CUR_DATA_DIR}/dev.final.${lang}
    cat ${CUR_DATA_DIR}/dt.newstest2014.tok.tc.bpe.${lang} > ${CUR_DATA_DIR}/test.final.${lang}
done
########### -- task specific -- ###########

# prepare dictionary for nematus
for lang in ${SRC} ${TRG}; do
    echo "Build dictionary for ${lang}"
    python2 ${RUNNING_DIR}/nematus_build_vocab.py \
        < "${CUR_DATA_DIR}/train.final.${lang}" > "${CUR_DATA_DIR}/vocab.${lang}.json"
done
