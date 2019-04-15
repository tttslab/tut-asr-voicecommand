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
import matplotlib.pyplot as plt

home_dir = os.path.expanduser('~')
logging.basicConfig(level=logging.INFO)

# Parameter setting
# example of tuning by argparser: python train.py --IN_SIZE=40 --USE_CMVN
parser = argparse.ArgumentParser()
parser.add_argument('--IN_SIZE',     type=int, default=13)             # input mfcc dim
parser.add_argument('--HIDDEN_SIZE', type=int, default=128)            # unit num of layer
parser.add_argument('--NUM_STACK',   type=int, default=5)              # number of layers
parser.add_argument('--DROPOUT',     type=float, default=0.5)
parser.add_argument('--USE_CMVN',    action='store_true')              # whether applying CMVN, default is False
parser.add_argument('--MAX_ITERATION', type=int, default=1000000)
parser.add_argument('--MAX_EPOCH',   type=int, default=5)
parser.add_argument('--BATCH_SIZE',  type=int, default=256)
args = parser.parse_args()

IN_SIZE       = args.IN_SIZE
NUM_CLASS     = 10
HIDDEN_SIZE   = args.HIDDEN_SIZE
NUM_STACK     = args.NUM_STACK
DROPOUT       = args.DROPOUT
USE_CMVN      = args.USE_CMVN
MAX_ITERATION = args.MAX_ITERATION
MAX_EPOCH     = args.MAX_EPOCH
BATCH_SIZE    = args.BATCH_SIZE

# Build up model and batch generator
device      = 'cuda' if torch.cuda.is_available() else 'cpu'  # check available gpu
model       = models.Classifier(IN_SIZE, NUM_CLASS, HIDDEN_SIZE, NUM_STACK, DROPOUT).to(device) # build up model
loss_fun    = nn.CrossEntropyLoss() # define CE as loss function (objective function)
optimizer   = torch.optim.Adam(model.parameters()) # define optimizer (choosed adam here, you can try others as well)
batch_train = utils.Batch_generator('training',   BATCH_SIZE) # batch generator
batch_test  = utils.Batch_generator('testing',    BATCH_SIZE)
batch_valid = utils.Batch_generator('validation', BATCH_SIZE)

# print out settings
logging.info('Batch_size: {}'.format(BATCH_SIZE))
logging.info('Max epoch: {}'.format(MAX_EPOCH))
logging.info('Max iteration: {}'.format(MAX_ITERATION))
logging.info('Hidden size: {}'.format(HIDDEN_SIZE))
logging.info('Num stack: {}'.format(NUM_STACK))
logging.info('Use cmvn: {}'.format(USE_CMVN))

# Training part
now_epoch   = 1
total_num   = 0 # total number of used data
correct_num = 0 # number of corrected prediction
acc_plt = []
epoch_plt = []
for iteration in range(1, MAX_ITERATION+1):
    model.train() # train the model
    inputs, labels, lengths, epoch = next(batch_train) # generate next batch
    if USE_CMVN:
        inputs  = utils.Apply_cmvn(torch.from_numpy(inputs).to(device)) # use cmvn
    else:
        inputs  = torch.from_numpy(inputs).to(device)
    inputs  = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True) #  pack the padded sequence (remove the redundancy padding)
    labels  = torch.from_numpy(labels).to(device) # load label
    
    outputs = model(inputs)
    loss    = loss_fun(outputs, labels) # compute loss
    optimizer.zero_grad() # clear gradient for all optimized tensor (initialize with 0)
    loss.backward() # gradient backpropagation
    optimizer.step() # update parameters

    total_num   += len(outputs) # compute total num
    correct_num += torch.bincount(torch.abs(torch.argmax(outputs, dim=1) - labels))[0] # compute corrected num

    logging.info('epoch: {}\titer: {}\tloss: {:.04f}'.format(now_epoch, iteration, loss.item()))
    if now_epoch < epoch:
        logging.info('training_accuracy: {:.04f}'.format(correct_num.float() / total_num))
        now_epoch   = epoch
        correct_num = 0
        total_num   = 0

        with torch.no_grad():
            model.eval()
            while True:
                inputs, labels, lengths, epoch = next(batch_valid)

                if USE_CMVN:
                    inputs  = utils.Apply_cmvn(torch.from_numpy(inputs).to(device))
                else:
                    inputs  = torch.from_numpy(inputs).to(device)
                inputs  = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True)
                labels  = torch.from_numpy(labels).to(device)
                outputs = model(inputs)
                loss    = loss_fun(outputs, labels)
                total_num   += len(outputs)
                correct_num += torch.bincount(torch.abs(torch.argmax(outputs, dim=1) - labels))[0]
                if epoch == now_epoch: break
        logging.info('validation_accuracy: {:.04f}'.format(correct_num.float() / total_num))
        acc_plt.append(float("{:.04f}".format(correct_num.float() / total_num)))
        epoch_plt.append(now_epoch-1)
        #logging.info(acc_plt)
        #logging.info(epoch_plt)
        correct_num = 0
        total_num   = 0

    if MAX_EPOCH < now_epoch:
        break

#plt.plot(epoch_plt, acc_plt, "ro-")
#plt.xlabel("epoch")
#plt.ylabel("accuracy")
#plt.savefig("result.png")

logging.info('done')
os.makedirs(home_dir + '/e2e_asr/model', exist_ok=True) # make dir for model saving
torch.save(model.state_dict(), home_dir + '/e2e_asr/model/speech_commands.model') # save trained model
