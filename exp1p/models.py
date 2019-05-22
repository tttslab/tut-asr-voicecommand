import torch
import torch.nn as nn
import torch.nn.functional as F

# This model classifies input audio to some classes.
# The structure is stacked biLSTM followed by an inference layer.
# The inference layer receives context vector of the encoder.

# in_size:     Input feature dimension, typically corresponds with MFCC dimension.
# num_class:   The number of classes to infer.
# hidden_size: The number of one side of encoder-lstm nodes.
#              The encoder is stacked biLSTM, so encoder-outputs dimension is 2*hidden_size.
# num_stack:   How many lstms in the encoder are stacked. 
# dropout:     Dropout ratio. This is known to be effective for overfitting.
class Classifier(nn.Module):
    def __init__(self, in_size, num_class, hidden_size, num_stack, dropout):
        super(Classifier, self).__init__()
        self.encoder   = nn.LSTM(in_size, hidden_size, num_stack, batch_first=True, bidirectional=True, dropout=dropout)
        self.inferring = nn.Linear(hidden_size*2, num_class)

    def forward(self, inputs):
        hidden_state, (h, c) = self.encoder(inputs)

        # Concatenate forward and backward LSTM's context vector
        context              = torch.cat((h[-2,:], h[-1,:]), dim=1)
        return self.inferring(context)
