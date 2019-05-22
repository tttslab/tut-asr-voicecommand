#!/bin/bash 

if [ $# -ne 3 ]; then
	echo "usage: $0 model_dir data_root eval_name" 1>&2
	exit 1
fi

MODEL_DIR=${1%/}
DATA_ROOT=${2%/}
MFCC_ROOT=$DATA_ROOT/mfcc
EVAL_LIST=$DATA_ROOT/$3

python eval.py \
	--MFCC_ROOT $MFCC_ROOT \
	--EVAL_LIST $EVAL_LIST \
	--PARAM_FILE $MODEL_DIR/trained.model
