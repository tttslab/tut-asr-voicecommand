#!/bin/sh

echo "start"
dir=$HOME/e2e_asr
#cd $dir 
if [ ! -e "$dir" ] ;then
    mkdir $dir
fi
log=$dir/exp.log
if [ -e "$log" ] ;then
    echo "log exist, delete"
    rm $log
fi

qsub -g tga-egliteracy -o $log -e $log qsub.sh
