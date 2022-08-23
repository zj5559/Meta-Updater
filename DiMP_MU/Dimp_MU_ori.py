import cv2
import os
import torch
import numpy as np
import sys
sys.path.append(os.path.join('../utils/metric_net'))

from tcopt import tcopts
from metric_model import ft_net
from torch.autograd import Variable
from me_sample_generator import *

# Dimp
import argparse
from pytracking.libs.tensorlist import TensorList
from pytracking.utils.plotting import show_tensor
from pytracking.features.preprocessing import numpy_to_torch
env_path = os.path.join(os.path.dirname(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.evaluation import Tracker

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
from tracking_utils import compute_iou, show_res, process_regions
import ltr.admin.loading as ltr_loading


class Dimp_MU_ori_Tracker(object):
    def __init__(self, image, region, p=None, groundtruth=None):
        self.p = p
        self.i = 0
        if groundtruth is not None:
            self.groundtruth = groundtruth

        init_gt1 = [region.x, region.y, region.width, region.height]
        init_gt = [init_gt1[1], init_gt1[0], init_gt1[1]+init_gt1[3], init_gt1[0]+init_gt1[2]]  # ymin xmin ymax xmax

        self.last_gt = init_gt

        self.local_init(image, init_gt1)

        self.tc_init(self.p.model_dir)
        self.metric_init(image, np.array(init_gt1))
        self.dis_record = []
        self.state_record = []
        self.rv_record = []
        self.all_map = []

        local_state1, self.score_map, self.score_max, dis,update = self.local_track(image)
        self.local_Tracker.pos = torch.FloatTensor(
            [(self.last_gt[0] + self.last_gt[2] - 1) / 2, (self.last_gt[1] + self.last_gt[3] - 1) / 2])
        self.local_Tracker.target_sz = torch.FloatTensor(
            [(self.last_gt[2] - self.last_gt[0]), (self.last_gt[3] - self.last_gt[1])])

    def get_first_state(self):
        return self.score_map, self.score_max

    def metric_init(self, im, init_box):
        self.metric_model = ft_net(class_num=1120)
        path = '../utils/metric_net/metric_model/metric_model_ori.pt'
        self.metric_model.eval()
        self.metric_model = self.metric_model.cuda()
        self.metric_model.load_state_dict(torch.load(path))
        tmp = np.random.rand(1, 3, 107, 107)
        tmp = (Variable(torch.Tensor(tmp))).type(torch.FloatTensor).cuda()
        # get target feature
        self.metric_model(tmp)
        init_box = init_box.reshape((1, 4))
        anchor_region = me_extract_regions(im, init_box)
        anchor_region = process_regions(anchor_region)
        anchor_region = torch.Tensor(anchor_region)
        anchor_region = (Variable(anchor_region)).type(torch.FloatTensor).cuda()
        self.anchor_feature, _ = self.metric_model(anchor_region)

    def metric_eval(self, im, boxes, anchor_feature):
        box_regions = me_extract_regions(np.array(im), boxes)
        box_regions = process_regions(box_regions)
        box_regions = torch.Tensor(box_regions)
        box_regions = (Variable(box_regions)).type(torch.FloatTensor).cuda()
        box_features, class_result = self.metric_model(box_regions)

        ap_dist = torch.norm(anchor_feature - box_features, 2, dim=1).view(-1)
        return ap_dist

    def tc_init(self, model_dir):
        path = '../models/DiMP_MU_ori.pth.tar'
        self.tc_model, _ = ltr_loading.load_network(path)
        self.tc_model = self.tc_model.cuda()
        self.tc_model.eval()

        tmp1 = np.random.rand(20, 1, 19, 19)
        tmp1 = (Variable(torch.Tensor(tmp1))).type(torch.FloatTensor).cuda()
        tmp2 = np.random.rand(1, 20, 6)
        tmp2 = (Variable(torch.Tensor(tmp2))).type(torch.FloatTensor).cuda()
        # get target feature
        self.tc_model(tmp2, tmp1)
    def local_init(self, image, init_bbox):
        local_tracker = Tracker('dimp', 'dimp50')
        params = local_tracker.get_parameters()

        debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = local_tracker.name
        params.param_name = local_tracker.parameter_name

        self.local_Tracker = local_tracker.tracker_class(params)
        init_box = dict()
        init_box['init_bbox'] = init_bbox
        self.local_Tracker.initialize(image, init_box)

    def local_track(self, image):
        state, score_map, test_x, scale_ind, sample_pos, sample_scales, flag, s = self.local_Tracker.track_updater(image)
        update_flag = flag not in ['not_found', 'uncertain']
        update = update_flag
        max_score = max(score_map.flatten())
        self.all_map.append(score_map)
        local_state = np.array(state).reshape((1, 4))
        # print(local_state)
        ap_dis = self.metric_eval(image, local_state, self.anchor_feature)
        ap_dis0=ap_dis.data.cpu().numpy()[0]
        self.dis_record.append(ap_dis0)
        h = image.shape[0]
        w = image.shape[1]
        self.state_record.append([local_state[0][0] / w, local_state[0][1] / h,
                                  (local_state[0][0] + local_state[0][2]) / w,
                                  (local_state[0][1] + local_state[0][3]) / h])
        self.rv_record.append(max_score)
        if len(self.state_record) >= tcopts['time_steps']:
            dis = np.array(self.dis_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
            rv = np.array(self.rv_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
            state_tc = np.array(self.state_record[-tcopts["time_steps"]:])
            map_input = np.array(self.all_map[-tcopts["time_steps"]:])
            map_input = np.reshape(map_input, [tcopts['time_steps'], 1, 19, 19])
            X_input = np.concatenate((state_tc, rv, dis), axis=1)
            X_input = X_input[np.newaxis, :]
            X_input = torch.tensor(X_input, dtype=torch.float, device='cuda')
            map_input = torch.tensor(map_input, dtype=torch.float, device='cuda')
            logits, _, _ = self.tc_model(X_input, map_input)
            update = logits[0][0] < logits[0][1]
            update=update.item()

        hard_negative = (flag == 'hard_negative')
        learning_rate = getattr(self.local_Tracker.params, 'hard_negative_learning_rate', None) if hard_negative else None

        if update:
            # Get train sample
            train_x = test_x[scale_ind:scale_ind+1, ...]

            # Create target_box and label for spatial sample
            target_box = self.local_Tracker.get_iounet_box(self.local_Tracker.pos, self.local_Tracker.target_sz,
                                                           sample_pos[scale_ind, :], sample_scales[scale_ind])

            # Update the classifier model
            self.local_Tracker.update_classifier(train_x, target_box, learning_rate, s[scale_ind,...])
        self.last_gt = [state[1], state[0], state[1]+state[3], state[0]+state[2]]
        return state, score_map, max_score, ap_dis.data.cpu().numpy()[0],update

    def locate(self, image):

        # Convert image
        im = numpy_to_torch(image)
        self.local_Tracker.im = im  # For debugging only

        # ------- LOCALIZATION ------- #

        # Get sample
        sample_pos = self.local_Tracker.pos.round()
        sample_scales = self.local_Tracker.target_scale * self.local_Tracker.params.scale_factors
        test_x = self.local_Tracker.extract_processed_sample(im, self.local_Tracker.pos, sample_scales, self.local_Tracker.img_sample_sz)

        # Compute scores
        scores_raw = self.local_Tracker.apply_filter(test_x)
        translation_vec, scale_ind, s, flag = self.local_Tracker.localize_target(scores_raw)
        return translation_vec, scale_ind, s, flag, sample_pos, sample_scales, test_x

    def local_update(self, sample_pos, translation_vec, scale_ind, sample_scales, s, test_x, update_flag=None):

        # Check flags and set learning rate if hard negative
        if update_flag is None:
            update_flag = self.flag not in ['not_found', 'uncertain']
        hard_negative = (self.flag == 'hard_negative')
        learning_rate = self.local_Tracker.params.hard_negative_learning_rate if hard_negative else None

        if update_flag:
            # Get train sample
            train_x = TensorList([x[scale_ind:scale_ind + 1, ...] for x in test_x])

            # Create label for sample
            train_y = self.local_Tracker.get_label_function(sample_pos, sample_scales[scale_ind])

            # Update memory
            self.local_Tracker.update_memory(train_x, train_y, learning_rate)

        # Train filter
        if hard_negative:
            self.local_Tracker.filter_optimizer.run(self.local_Tracker.params.hard_negative_CG_iter)
        elif (self.local_Tracker.frame_num - 1) % self.local_Tracker.params.train_skipping == 0:
            self.local_Tracker.filter_optimizer.run(self.local_Tracker.params.CG_iter)

    def tracking(self, image):
        self.i += 1
        local_state1, self.score_map, score_max, dis,update = self.local_track(image)
        if self.groundtruth is None:
            iou = 1
        elif self.groundtruth.shape[0] == 1:
            iou = 1
        else:
            gt_err = self.groundtruth[self.i, 2] < 3 or self.groundtruth[self.i, 3] < 3
            gt_nan = any(np.isnan(self.groundtruth[self.i]))
            if gt_err:
                iou = -1
            elif gt_nan:
                iou = 0
            else:
                iou = compute_iou(self.groundtruth[self.i], local_state1)

        width = self.last_gt[3] - self.last_gt[1]
        height = self.last_gt[2] - self.last_gt[0]

        if self.p.visualization:
            show_res(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), np.array(self.last_gt, dtype=np.int32), '2',
                     groundtruth=self.groundtruth, update=update,
                     frame_id=self.i, score=max(self.score_map.flatten()))

        return [float(self.last_gt[1]), float(self.last_gt[0]), float(width),
                float(height)], self.score_map, iou, score_max, dis,dis
