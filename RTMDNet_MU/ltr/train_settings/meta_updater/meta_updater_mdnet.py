import torch.nn as nn
import torch.optim as optim
from ltr.data import sampler_mu_mdnet, LTRLoader
from ltr.models.tracking import tcNet
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.tcopt import tcopts
import numpy as np
import os
import sys
sys.path.append('../..')
from local_path import base_path
def load_training_data(pos_name, neg_name):
    pos_data = np.load(pos_name, allow_pickle=True)
    neg_data = np.load(neg_name, allow_pickle=True)
    return pos_data, neg_data
def prepare_test_data(Dataset, seq_len, mode=None):
    base_dir = tcopts['train_data_dir']
    data_dir = os.path.join(base_dir, Dataset)
    train_list = os.listdir(data_dir)
    train_list.sort()
    np.random.shuffle(train_list)

    testing_set_dir = '../../utils/testing_set.txt'
    testing_set = list(np.loadtxt(testing_set_dir, dtype=str))
    if mode == 'test':
        print('test data')
        train_list = [vid for vid in train_list if vid[:-4] in testing_set]
    elif mode == 'train':
        print('train data')
        train_list = [vid for vid in train_list if vid[:-4] not in testing_set]
    else:
        print("all data")
    pos_data, neg_data = prepare_data(data_dir, seq_len, train_list)
    np.save('test_neg_data.npy', np.array(neg_data))
    np.save('test_pos_data.npy', np.array(pos_data))
    return pos_data, neg_data

def prepare_train_data(Dataset, seq_len, mode=None):
    base_dir = tcopts['train_data_dir']
    data_dir = os.path.join(base_dir, Dataset)
    train_list = os.listdir(data_dir)
    train_list.sort()
    np.random.shuffle(train_list)

    testing_set_dir = '../../utils/testing_set.txt'
    testing_set = list(np.loadtxt(testing_set_dir, dtype=str))
    if mode == 'test':
        print('test data')
        train_list = [vid for vid in train_list if vid[:-4] in testing_set]
    elif mode == 'train':
        print('train data')
        train_list = [vid for vid in train_list if vid[:-4] not in testing_set]
    else:
        print("all data")

    pos_data, neg_data = prepare_data(data_dir, seq_len,train_list)
    np.save('neg_data.npy', np.array(neg_data))
    np.save('pos_data.npy', np.array(pos_data))
    return pos_data, neg_data

def prepare_data(data_dir, seq_len, train_list):
    pos_data = []
    neg_data = []
    sampling_interval = tcopts['sampling_interval']
    # video
    for id, video in enumerate(train_list):
        print(str(id) + ':' + video)
        txt_tmp = np.loadtxt(os.path.join(data_dir, train_list[id]), delimiter=',')
        loss_list = np.where(txt_tmp[:, 5] == 0)[0]
        for i in range((len(txt_tmp) - seq_len)//sampling_interval):
            data_tmp = txt_tmp[sampling_interval*i:sampling_interval*i + seq_len]
            loss_index = np.concatenate([np.where(data_tmp[:, 5] == -1)[0], np.where(data_tmp[:, 5] == 0)[0]])
            if data_tmp[-1, 5] > tcopts['pos_thr']:
                # pos data
                pos_data.append(data_tmp)
            elif data_tmp[-1, 5] == 0:
                neg_data.append(data_tmp)
    return pos_data, neg_data

def run(settings):
    if not os.path.exists(tcopts['lstm_train_dir']):
        os.mkdir(tcopts['lstm_train_dir'])
    settings.description = 'Meta-updater with default settings.'
    settings.print_interval = 1

    prepare_train_data('', tcopts['time_steps'], mode='train')
    prepare_test_data('', tcopts['time_steps'], mode='test')

    pos_data, neg_data = load_training_data('pos_data.npy', 'neg_data.npy')
    dataset_train = sampler_mu_mdnet.MUMdnetSampler(pos_data,neg_data,samples_per_epoch=50000)
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=tcopts['batch_size'], num_workers=tcopts['num_threads'],
                             shuffle=True, drop_last=True, stack_dim=0)

    pos_data, neg_data = load_training_data('test_pos_data.npy', 'test_neg_data.npy')
    dataset_val = sampler_mu_mdnet.MUMdnetSampler(pos_data, neg_data, samples_per_epoch=10000)
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=500, num_workers=tcopts['num_threads'],
                             shuffle=False, drop_last=True, stack_dim=0)

    # Create network and actor
    net = tcNet.mu_model()

    # Wrap the network for multi GPU training
    if tcopts['multi_gpu']:
        net = MultiGPU(net, dim=1)

    objective = tcNet.mu_loss()

    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    actor = actors.MUActor_mdnet(net=net, objective=objective)

    param_dicts = [
        {"params": [p for n, p in net.named_parameters() if p.requires_grad]}
    ]
    optimizer = optim.SGD(param_dicts, lr=tcopts['lstm_initial_lr'],momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=tcopts['lstm_decay_steps'], gamma=tcopts['lr_decay_factor'])

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(50, load_latest=True, fail_safe=True)
