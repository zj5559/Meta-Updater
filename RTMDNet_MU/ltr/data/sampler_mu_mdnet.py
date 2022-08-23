import random
import os
import torch.utils.data
from pytracking import TensorDict
import numpy as np
from ltr.tcopt_mdnet import tcopts

class TrackingSampler(torch.utils.data.Dataset):
    def __init__(self, pos_data, neg_data,samples_per_epoch):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_test_frames - Number of test frames to sample.
            num_train_frames - Number of train frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.pos_num = len(pos_data)
        self.neg_num = len(neg_data)
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        pos_id = np.random.randint(0, self.pos_num)
        neg_id = np.random.randint(0, self.neg_num)

        pos_input = self.pos_data[pos_id][:,tcopts['lstm_input']]#[20,8]
        neg_input= self.neg_data[neg_id][:,tcopts['lstm_input']]

        pos_input[:, -1] = (1.0 / (pos_input[:, -1] + 0.0001))
        neg_input[:, -1] = (1.0 / (neg_input[:, -1] + 0.0001))

        pos_input[:, -2] = (1.0 / (pos_input[:, -2] + 0.0001))
        neg_input[:, -2] = (1.0 / (neg_input[:, -2] + 0.0001))

        pos_input = torch.tensor(pos_input, dtype=torch.float, device=tcopts['device'])
        neg_input = torch.tensor(neg_input, dtype=torch.float, device=tcopts['device'])
        data = TensorDict({'pos_input': pos_input,
                           'neg_input': neg_input})

        return data


class MUMdnetSampler(TrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, pos_data, neg_data,samples_per_epoch):
        super().__init__(pos_data, neg_data,samples_per_epoch)
