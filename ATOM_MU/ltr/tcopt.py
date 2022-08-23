from collections import OrderedDict

tcopts = OrderedDict()

# training batch
tcopts['device']='cuda'
tcopts['batch_size'] = 32  # peer gpu
tcopts['lr_decay_factor'] = 0.1
tcopts['num_threads'] = 8
tcopts['capacity'] = 32
tcopts['multi_gpu']=False

# lstm
tcopts['lstm_train_dir'] = '/media/zj/4T-1/model/ltmu_new2/ATOM_MU'
tcopts['train_data_dir'] = '/home/zj/tracking/LTMU_Expansion/LTMU-master/ATOM_MU/results_old/ATOM/lasot/train_data'
tcopts['pos_thr'] = 0.5
tcopts['lstm_initial_lr'] = 1e-3
tcopts['lstm_decay_steps'] = 30#50000
tcopts['lstm_input'] = [0, 1, 2, 3, 6, 7,8]
tcopts['sampling_interval'] = 4
tcopts['lstm_num_input'] = len(tcopts['lstm_input'])
tcopts['lstm_num_classes'] = 2
tcopts['time_steps'] = 20
tcopts['start_frame'] = 20
tcopts['map_units'] = 4
