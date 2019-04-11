#!/bin/bash 

if [ $# -ne 3 ]; then
	echo "usage: $0 model_dir data_root train_name" 1>&2
	exit 1
fi

MODEL_DIR=${1%/}
DATA_ROOT=${2%/}
MFCC_ROOT=$DATA_ROOT/mfcc
TRAIN_LIST=$DATA_ROOT/$3
VALID_LIST=$DATA_ROOT/valid.txt

if [ -e $MODEL_DIR/done ]; then
	echo "model is already trained. Skip"
	exit 0
fi

mkdir -p $MODEL_DIR || exit 1
python train.py \
	--MFCC_ROOT $MFCC_ROOT \
	--TRAIN_LIST $TRAIN_LIST \
	--VALID_LIST $VALID_LIST \
	--SAVE_FILE $MODEL_DIR/trained.model \
	--MAX_EPOCH=5 \
	|| exit 1 ## Train NN
touch $MODEL_DIR/done
