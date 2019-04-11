#!/bin/sh

echo 'start'
date
#./data_download.sh ## Download speech_commands dataset
python ./preprocess.py  ## Generate MFCC feature (set as stage0 later)
echo 'start training'
date
python ./train.py --MAX_EPOCH=100 ## Train NN 
echo 'start eval'
date
python ./eval.py
date
