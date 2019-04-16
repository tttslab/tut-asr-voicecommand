#!/bin/bash

DATA_ROOT=data
MODEL_DIR=model

echo 'start preparing'
date
./prep.sh $DATA_ROOT >& prep.log
echo 'start training'
date
./train.sh $MODEL_DIR $DATA_ROOT train2.txt >& train.log
#./train.sh $MODEL_DIR $DATA_ROOT train20.txt >& train.log
#./train.sh $MODEL_DIR $DATA_ROOT train40.txt >& train.log
#./train.sh $MODEL_DIR $DATA_ROOT train60.txt >& train.log
#./train.sh $MODEL_DIR $DATA_ROOT train80.txt >& train.log
#./train.sh $MODEL_DIR $DATA_ROOT train.txt >& train.log
echo 'start evaluating'
date
./eval.sh $MODEL_DIR $DATA_ROOT eval.txt >& eval.log
date
