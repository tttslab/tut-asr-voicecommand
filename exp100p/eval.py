import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
import utils
import sys
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO)

# Parameter setting
# example of tuning by argparser: python train.py --IN_SIZE=40 --USE_CMVN
parser = argparse.ArgumentParser()
parser.add_argument('--IN_SIZE',     type=int, default=13)             # input mfcc dim
parser.add_argument('--HIDDEN_SIZE', type=int, default=128)            # unit num of layer
parser.add_argument('--NUM_STACK',   type=int, default=5)              # number of layers
parser.add_argument('--USE_CMVN',    action='store_true')              # whether applying CMVN, default is False
parser.add_argument('--BATCH_SIZE',  type=int, default=256)
parser.add_argument('--PARAM_FILE',  type=str)                         # parameter of trained model
parser.add_argument('--EVAL_LIST',  type=str)                          # evaluation file list
parser.add_argument('--MFCC_ROOT',  type=str)                          # root path of mfcc feature files
args = parser.parse_args()

IN_SIZE       = args.IN_SIZE
NUM_CLASS     = 10
HIDDEN_SIZE   = args.HIDDEN_SIZE
NUM_STACK     = args.NUM_STACK
USE_CMVN      = args.USE_CMVN
BATCH_SIZE    = args.BATCH_SIZE
PARAM_FILE    = args.PARAM_FILE
EVAL_LIST     = args.EVAL_LIST
MFCC_ROOT     = args.MFCC_ROOT

# Build up model and batch generator
device      = 'cuda' if torch.cuda.is_available() else 'cpu'   # check available gpu
model       = models.Classifier(IN_SIZE, NUM_CLASS, HIDDEN_SIZE, NUM_STACK, 0.0).to(device) # build model (same structure as trained model)
model.load_state_dict(torch.load(PARAM_FILE)) # load parameters from trained model
batch_test  = utils.Batch_generator(MFCC_ROOT, EVAL_LIST, BATCH_SIZE) # data batch generator for evaluation data

# Print out setting
logging.info('Batch_size: {}'.format(BATCH_SIZE))
logging.info('Hidden size: {}'.format(HIDDEN_SIZE))
logging.info('Num stack: {}'.format(NUM_STACK))
logging.info('Use cmvn: {}'.format(USE_CMVN))

# Training part
with torch.no_grad(): # disable gradient calculation, reduce memory consumption
    model.eval()
    total_num   = 0 # total num of test data
    correct_num = 0 # corrected prediction num
    while True:
        inputs, labels, lengths, epoch = next(batch_test) # generate a data batch
        if USE_CMVN:
            inputs  = utils.Apply_cmvn(torch.from_numpy(inputs).to(device)) # use cmvn
        else:
            inputs  = torch.from_numpy(inputs).to(device)
        inputs  = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True) # pack the padded sequence (remove the redundancy padding)
        labels  = torch.from_numpy(labels).to(device)
        outputs = model(inputs)
        total_num   += len(outputs)
        correct_num += torch.bincount(torch.abs(torch.argmax(outputs, dim=1) - labels))[0] # compute the number of corrected prediction
        if epoch == 2: break
    score = correct_num.float() / total_num
    logging.info('accuracy: {:.04f}'.format(score))
    with open('score.txt', 'w') as f:
        f.write('recognition rate = {:.1%}\n'.format(score))

logging.info('done')
