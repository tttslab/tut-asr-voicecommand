#!/bin/bash

#DATA_ROOT=data
DATA_ROOT=/gs/hs0/tga-tslecture/data/tut-asr-voicecommand/data
MODEL_DIR=model

echo 'start preparing'
date
./prep.sh $DATA_ROOT >& prep.log

echo 'start training'
date

start_time=`date +%s`


#./train.sh $MODEL_DIR $DATA_ROOT train1p.txt >& train.log
#./train.sh $MODEL_DIR $DATA_ROOT train20p.txt >& train.log
#./train.sh $MODEL_DIR $DATA_ROOT train60p.txt >& train.log
./train.sh $MODEL_DIR $DATA_ROOT train100p.txt >& train.log

end_time=`date +%s`
time=$((end_time - start_time))
echo "${time} (sec)" >& train.time.log

echo 'start evaluating'
date
./eval.sh $MODEL_DIR $DATA_ROOT eval.txt >& eval.log
date
