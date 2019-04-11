import torch
import numpy as np
import random
import os

command_list   = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

def Apply_cmvn(inputs): # apply cepstral mean and variance normalization
    batch_size, time, dim = inputs.shape
    mu    = torch.mean(inputs, dim=1).repeat(1, time).reshape(batch_size, time, dim)
    sigma = torch.pow(torch.mean(torch.pow(inputs, 2), dim=1).repeat(1, time).reshape(batch_size, time, dim) - torch.pow(mu, 2), 0.5)
    return (inputs - mu) / sigma

def insert_index_descending_order(query, num_list):
    matching_list = list(filter(lambda x: x < query, num_list)) # list(filter(if x < query for x in num_list))
    if len(matching_list) == 0:
        return len(num_list)
    else:
        return num_list.index(matching_list[0])
    

def Batch_generator(mfcc_root, dataset, batch_size): # data batch generator
    datalist_txt = open(dataset, 'r')
    OUTDIR = mfcc_root + '/'

    datalist      = datalist_txt.read().strip().split('\n')
    shuffled_data = random.sample(datalist, len(datalist))
    datalist_txt.close()
    epoch         = 1

    while True:
        data_batch   = np.array([], dtype=np.float32)
        label_batch  = []
        length_batch = []
        MAX_LEN      = 0
        for i in range(batch_size):
            sample  = shuffled_data.pop() # pop data from shuffled dataset
            label   = sample.split('/')[0]
            mfcc    = np.load(OUTDIR + sample)
            MAX_LEN = len(mfcc) if MAX_LEN < len(mfcc) else MAX_LEN # find max len in a batch
            index   = insert_index_descending_order(len(mfcc), length_batch) # insert data to get the decending sequence (for latter pack_padded_sequence)
            if i == 0:
                data_batch = np.asarray([mfcc])
            else:
                data_batch = np.pad(data_batch, ((0, 0), (0, MAX_LEN - data_batch.shape[1]), (0, 0)), mode='constant', constant_values=0)
                data_batch = np.insert(data_batch, index, np.pad(mfcc, ((0, MAX_LEN - len(mfcc)), (0, 0)), mode='constant', constant_values=0), axis=0)
            label_batch.insert(index, command_list.index(label)) # add to current batch
            length_batch.insert(index, len(mfcc))
        data_batch  = np.asarray(data_batch,  dtype=np.float32) # format change
        label_batch = np.asarray(label_batch, dtype=np.int64)

        if len(shuffled_data) < batch_size: # if remaining data (wait for pop into the batch) is not enough, do extension
            shuffled_data = random.sample(datalist, len(datalist)) + shuffled_data
            epoch        += 1

        yield data_batch, label_batch, length_batch, epoch
