import random
import os
import torch.utils.data
from pytracking import TensorDict
import numpy as np
from ltr.tcopt import tcopts

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

        map_tmp = np.load(os.path.join(tcopts['train_data_dir'], self.pos_data[pos_id][1]), allow_pickle=True)
        frame_index = self.pos_data[pos_id][0][:, 4]
        pos_map = np.reshape(map_tmp[np.array(frame_index, dtype='int16') - 1], [tcopts['time_steps'], 1, 19, 19])#[20,1,19,19]

        map_tmp = np.load(os.path.join(tcopts['train_data_dir'], self.neg_data[neg_id][1]), allow_pickle=True)
        frame_index = self.neg_data[neg_id][0][:, 4]
        neg_map = np.reshape(map_tmp[np.array(frame_index, dtype='int16') - 1], [tcopts['time_steps'], 1, 19, 19])

        pos_input = self.pos_data[pos_id][0][:,tcopts['lstm_input']]#[20,8]
        neg_input= self.neg_data[neg_id][0][:,tcopts['lstm_input']]

        # #z-score norm
        #lof_dis
        # pos_input[:,-1]=(pos_input[:,-1]-self.mean[2])/self.std[2]
        # neg_input[:, -1] = (neg_input[:,-1]-self.mean[2])/self.std[2]
        #
        # #ap_dis
        # pos_input[:,-2]=(pos_input[:,-2]-self.mean[1])/self.std[1]
        # neg_input[:, -2] = (neg_input[:,-2]-self.mean[1])/self.std[1]

        pos_input[:,-1]=(1.0/(pos_input[:,-1]+0.0001))
        neg_input[:, -1] = (1.0/(neg_input[:,-1]+0.0001))

        pos_input[:, -2] = (1.0 / (pos_input[:, -2]+0.0001))
        neg_input[:, -2] = (1.0 / (neg_input[:, -2]+0.0001))


        #min-max norm
        #lof_dis
        # pos_input[:,-1]=(pos_input[:,-1]-self.min[2])/(self.max[2]-self.min[2])
        # neg_input[:, -1] = (neg_input[:,-1]-self.min[2])/(self.max[2]-self.min[2])
        #
        # #ap_dis
        # pos_input[:,-2]=(pos_input[:,-2]-self.min[1])/(self.max[1]-self.min[1])
        # neg_input[:, -2] = (neg_input[:,-2]-self.min[1])/(self.max[1]-self.min[1])


        pos_input = torch.tensor(pos_input, dtype=torch.float, device=tcopts['device'])
        neg_input = torch.tensor(neg_input, dtype=torch.float, device=tcopts['device'])
        pos_map = torch.tensor(pos_map, dtype=torch.float, device=tcopts['device'])
        neg_map = torch.tensor(neg_map, dtype=torch.float, device=tcopts['device'])
        data = TensorDict({'pos_input': pos_input,
                           'neg_input': neg_input,
                           'pos_map': pos_map,
                           'neg_map': neg_map})

        return data


class MUSampler(TrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, pos_data, neg_data,samples_per_epoch):
        super().__init__(pos_data, neg_data,samples_per_epoch)
