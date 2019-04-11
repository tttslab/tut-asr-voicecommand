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
python preprocess.py --WAVE_DIR $DATA_ROOT/wave --OUT_DIR $DATA_ROOT/mfcc || exit 1
touch $DATA_ROOT/done
