import cv2
import os
import torch
import numpy as np
import sys
sys.path.append('../utils/metric_net')
from metric_model import ft_net
from torch.autograd import Variable
from me_sample_generator import *
from tcopt import tcopts
env_path = os.path.join(os.path.dirname(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.evaluation.tracker import Tracker

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
from tracking_utils import compute_iou, show_res, process_regions
import ltr.admin.loading as ltr_loading

from sklearn.neighbors import LocalOutlierFactor
def lof_fit(data,k=5,method='l2'):
    clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', metric=method, contamination=0.1)
    clf.fit(data)
    return clf
def lof(predict, clf,k=5, method='l2',thresh=2):
    predict = -clf._score_samples(predict)
    result=predict<=thresh
    return predict[0],result[0]

class Stark_MU_Tracker(object):
    def __init__(self, image, region, p=None, groundtruth=None):

        self.p = p
        self.i = 0
        self.globalmode = True
        if groundtruth is not None:
            self.groundtruth = groundtruth
        else:
            self.groundtruth=None

        init_gt1 = [region.x, region.y, region.width, region.height]
        init_gt = [init_gt1[1], init_gt1[0], init_gt1[1]+init_gt1[3], init_gt1[0]+init_gt1[2]] # ymin xmin ymax xmax

        self.last_gt = init_gt
        self.metric_init(image, np.array(init_gt1),p.lof_thresh)
        self.tc_init(self.p.model_dir)

        self.local_init(image, init_gt1)

        self.dis_record = []
        self.state_record = []
        self.rv_record = []
        self.lof_dis_record = []
        local_state1, self.score_max, ap_dis,lof_dis,update = self.local_track(image)
        self.local_Tracker.pos = torch.FloatTensor(
            [(self.last_gt[0] + self.last_gt[2] - 1) / 2, (self.last_gt[1] + self.last_gt[3] - 1) / 2])
        self.local_Tracker.target_sz = torch.FloatTensor(
            [(self.last_gt[2] - self.last_gt[0]), (self.last_gt[3] - self.last_gt[1])])

    def get_first_state(self):
        return self.score_max

    def metric_init(self, im, init_box,lof_thresh):
        self.metric_model = ft_net(class_num=1120)
        path = '../utils/metric_net/metric_model/metric_model.pt'
        self.metric_model.eval()
        self.metric_model = self.metric_model.cuda()
        self.metric_model.load_state_dict(torch.load(path))
        tmp = np.random.rand(1, 3, 107, 107)
        tmp = (Variable(torch.Tensor(tmp))).type(torch.FloatTensor).cuda()
        # get target feature
        self.metric_model(tmp)
        init_box = init_box.reshape((1, 4))
        with torch.no_grad():
            self.anchor_feature=self.get_metric_feature(im,init_box)

        pos_generator = SampleGenerator('gaussian', np.array([im.shape[1], im.shape[0]]), 0.1, 1.3)
        gt_pos_examples = pos_generator(init_box[0].astype(np.float32), 20, [0.7, 1])
        gt_iou = 0.7
        print(gt_pos_examples.shape[0])
        while gt_pos_examples.shape[0] < 10:
            gt_iou = gt_iou - 0.05
            gt_pos_examples = pos_generator(init_box[0].astype(np.float32), 20, [gt_iou, 1])
            print(gt_pos_examples.shape[0])
        with torch.no_grad():
            self.gt_pos_features=self.get_metric_feature(im,gt_pos_examples).cpu().detach().numpy()

        self.clf = lof_fit(self.gt_pos_features, k=5)
        self.lof_thresh = lof_thresh#2.5#2.1
    def get_metric_feature(self,im,box):
        anchor_region = me_extract_regions(im, box)
        anchor_region = process_regions(anchor_region)
        anchor_region = torch.Tensor(anchor_region)
        anchor_region = (Variable(anchor_region)).type(torch.FloatTensor).cuda()
        anchor_feature, _ = self.metric_model(anchor_region)
        return anchor_feature

    def metric_eval(self, im, boxes):
        box_features = self.get_metric_feature(np.array(im), boxes)
        ap_dist = torch.norm(self.anchor_feature - box_features, 2, dim=1).view(-1)
        lof_score,success= lof(box_features.cpu().detach().numpy(), self.clf, k=5, thresh=self.lof_thresh)
        return ap_dist.data.cpu().numpy()[0],lof_score,success

    def tc_init(self, model_dir):
        path = '../models/Stark_MU.pth.tar'
        self.tc_model, _ = ltr_loading.load_network(path)
        self.tc_model = self.tc_model.cuda()
        self.tc_model.eval()

        tmp2 = np.random.rand(1, 20, 7)
        tmp2 = (Variable(torch.Tensor(tmp2))).type(torch.FloatTensor).cuda()
        # get target feature
        self.tc_model(tmp2)

    def local_init(self, image, init_bbox):
        local_tracker = Tracker('stark_st', 'baseline_R101')
        params = local_tracker.get_parameters()

        debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = local_tracker.name
        params.param_name = local_tracker.parameter_name

        self.local_Tracker = local_tracker.tracker_class(params,'lasot')
        init_box = dict()
        init_box['init_bbox'] = init_bbox
        self.local_Tracker.initialize(image, init_box)
        self.local_Tracker.update_intervals=self.p.update_interval

    def local_track(self, image):
        state, max_score = self.local_Tracker.track_update(image)
        self.last_gt = [state[1], state[0], state[1]+state[3], state[0]+state[2]]
        local_state = np.array(state).reshape((1, 4))
        ap_dis,lof_dis1,success = self.metric_eval(image, local_state)

        # self.dis_record.append((ap_dis-self.mean[1])/self.std[1])
        # self.lof_dis_record.append((lof_dis1-self.mean[2])/self.std[2])
        self.dis_record.append(1.0/(ap_dis+0.0001))
        self.lof_dis_record.append(1.0 / (lof_dis1+0.0001))
        h = image.shape[0]
        w = image.shape[1]
        self.state_record.append([local_state[0][0] / w, local_state[0][1] / h,
                                  (local_state[0][0] + local_state[0][2]) / w,
                                  (local_state[0][1] + local_state[0][3]) / h])
        self.rv_record.append(max_score)
        if len(self.state_record) >= tcopts['time_steps']:
            dis = np.array(self.dis_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
            lof_dis = np.array(self.lof_dis_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
            rv = np.array(self.rv_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
            state_tc = np.array(self.state_record[-tcopts["time_steps"]:])
            X_input = np.concatenate((state_tc, rv, dis,lof_dis), axis=1)
            X_input = X_input[np.newaxis, :]
            X_input = torch.tensor(X_input, dtype=torch.float, device='cuda')
            logits, _= self.tc_model(X_input)
            update = logits[0][0] < logits[0][1]
        else:
            update=success
        self.local_Tracker.update(image,update,max_score)

        return state, max_score, ap_dis,lof_dis1,update

    def tracking(self, image):
        self.i += 1
        local_state1, score_max, ap_dis,lof_dis,update = self.local_track(image)
        if self.groundtruth is None:
            iou=1
        elif self.groundtruth.shape[0]==1:
            iou=1
        else:
            gt_err = self.groundtruth[self.i, 2] < 3 or self.groundtruth[self.i, 3] < 3
            gt_nan = any(np.isnan(self.groundtruth[self.i]))
            if gt_err:
                iou = -1
            elif gt_nan:
                iou = 0
            else:
                iou = compute_iou(self.groundtruth[self.i], local_state1)
        # iou=1
        ##------------------------------------------------------##

        # self.local_update(sample_pos, translation_vec, scale_ind, sample_scales, s, test_x)

        width = self.last_gt[3] - self.last_gt[1]
        height = self.last_gt[2] - self.last_gt[0]

        if self.p.visualization:
            show_res(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), np.array(self.last_gt, dtype=np.int32), '2',
                     groundtruth=self.groundtruth,
                     frame_id=self.i, score=max(self.score_map.flatten()))

        return [float(self.last_gt[1]), float(self.last_gt[0]), float(width),
                float(height)], iou, score_max, ap_dis,lof_dis,update

