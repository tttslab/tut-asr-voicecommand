#!/bin/sh

# use this script to use mkgraph.py in tsubame's default env, wrote by peng
# usage
# ./mkgraph.sh 'x_label' 'x_values' 'y_label' 'y_values' 'save_file_name'
# e.g.
# ./mkgrapsh.sh 'data(%)' '20 40 60 80' 'accuracy(%)' '56.4 60.8 78.5 92.5' 'result.png'

export PATH="/gs/hs0/tga-egliteracy/egs/e2e-asr/miniconda3/envs/mnist/bin:$PATH"
python mkgraph.py -xl $1 -x $2 -yl $3 -y $4 -f $5
