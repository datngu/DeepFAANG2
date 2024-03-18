import math

import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_targets, sequence_length = 1000):
        super(Model, self).__init__()
        self.conv_pool_drop_1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(0.2))

        self.bdlstm = nn.LSTM(input_size=320, hidden_size=320, num_layers=1, batch_first=True, bidirectional=True)

        self.dropout_2 = nn.Dropout(0.5)

        self.dense_1 = nn.Sequential(
            nn.Linear(48000, 925),
            nn.ReLU())

        self.dense_2 = nn.Sequential(
            nn.Linear(925, n_targets),
            )
            

    def forward(self, inputs, training=None, mask=None, **kwargs):
        temp = self.conv_pool_drop_1(inputs)
        temp = temp.transpose(1, 2)

        temp, _ = self.bdlstm(temp)

        temp = self.dropout_2(temp)

        temp = temp.reshape(temp.shape[0], -1)
       
        temp = self.dense_1(temp)

        output = self.dense_2(temp)
        return output