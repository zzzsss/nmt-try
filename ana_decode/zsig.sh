#!/usr/bin/env bash

# significant test with paired_bootstrap_resampling_bleu_v13a.pl

set -v

SRC=$1
REF=$2
BASE=$3
MINE=$4

echo "SRC=$1, REF=$2, BASE=$3, MINE=$4"

perl ./wrap-xml.perl en $SRC zbase < $BASE > _base.sgm
perl ./wrap-xml.perl en $SRC zmine < $MINE > _mine.sgm
> _base.bleu.stats_file
> _mine.bleu.stats_file
perl ./mteval-v13a-sig_t2.pl -r $REF -t _base.sgm -s $SRC -f _base.bleu.stats_file
perl ./mteval-v13a-sig_t2.pl -r $REF -t _mine.sgm -s $SRC -f _mine.bleu.stats_file
perl ./paired_bootstrap_resampling_bleu_v13a.pl _mine.bleu.stats_file _base.bleu.stats_file 1000 0.05
