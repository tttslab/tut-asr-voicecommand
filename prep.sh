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

./data_download.sh $DATA_ROOT/wave || exit 1 ## Download speech_commands dataset
python preprocess.py --WAVE_DIR $DATA_ROOT/wave --TXT_DIR $DATA_ROOT --MFCC_DIR $DATA_ROOT/mfcc || exit 1 ## Generate MFCC feature (set as stage0 later)

#shuffle and separate training file list
cat $DATA_ROOT/train.txt | shuf -o $DATA_ROOT/train.txt
head -n $((1*`cat $DATA_ROOT/train.txt | wc -l`/100)) $DATA_ROOT/train.txt > $DATA_ROOT/train1.txt
head -n $((1*`cat $DATA_ROOT/train.txt | wc -l`/5)) $DATA_ROOT/train.txt > $DATA_ROOT/train20.txt
head -n $((3*`cat $DATA_ROOT/train.txt | wc -l`/5)) $DATA_ROOT/train.txt > $DATA_ROOT/train60.txt

touch $DATA_ROOT/done
