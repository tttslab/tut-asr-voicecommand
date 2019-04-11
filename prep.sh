#!/bin/bash 

if [ $# -ne 1 ]; then
	echo "usage: $0 data_root" 1>&2
	exit 1
fi

DATA_ROOT=${1%/}

if [ -e $DATA_ROOT/done ]; then
	echo "data is already prepared. Skip"
	exit 0
fi

./data_download.sh $DATA_ROOT/wave || exit 1
python preprocess.py --WAVE_DIR $DATA_ROOT/wave --TXT_DIR $DATA_ROOT --MFCC_DIR $DATA_ROOT/mfcc || exit 1

shuf $DATA_ROOT/train.txt | split -n l/1/5 - $DATA_ROOT/train20.txt
cat $DATA_ROOT/train.txt | shuf -o $DATA_ROOT/train.txt
head -n $((1*`cat data/train.txt | wc -l`/5)) $DATA_ROOT/train.txt > $DATA_ROOT/train20.txt
head -n $((2*`cat data/train.txt | wc -l`/5)) $DATA_ROOT/train.txt > $DATA_ROOT/train40.txt
head -n $((3*`cat data/train.txt | wc -l`/5)) $DATA_ROOT/train.txt > $DATA_ROOT/train60.txt
head -n $((4*`cat data/train.txt | wc -l`/5)) $DATA_ROOT/train.txt > $DATA_ROOT/train80.txt

touch $DATA_ROOT/done