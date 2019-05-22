#!/bin/bash

if [ $# -ne 1 ]; then
	echo "usage: $0 data_dir" 1>&2
	exit 1
fi

DATA_DIR=${1%/}

mkdir -p $DATA_DIR
echo 'start data downloading'
wget -q http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz -O $DATA_DIR.tar.gz
echo 'start expanding'
tar xzf $DATA_DIR.tar.gz -C $DATA_DIR
