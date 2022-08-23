import cv2
import os
import torch
import numpy as np
import sys
import tqdm
sys.path.append(os.path.join('lib'))
sys.path.append(os.path.join('../utils/metric_net'))
from metric_model import ft_net
from torch.autograd import Variable
from me_sample_generator import *

from easydict import EasyDict as edict
from tracker.ocean import Ocean
from tracker.online import ONLINE
import models.models as models
from lib.utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou

env_path = os.path.join(os.path.dirname(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
from tracking_utils import compute_iou, show_res, process_regions

from sklearn.neighbors import LocalOutlierFactor
def lof_fit(data,k=5,method='l2'):
    clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', metric=method, contamination=0.1)
    clf.fit(data)
    return clf
def lof(predict, clf,k=5, method='l2',thresh=2):
    predict = -clf._score_samples(predict)
    result=predict<=thresh
    return predict[0],result[0]

class Ocean_Tracker(object):
    def __init__(self, image_ori, region, p=None, groundtruth=None):
        image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
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
        self.metric_init(image, np.array(init_gt1))

        self.local_init(image_ori, init_gt1)

        local_state1, self.score_map,self.score_pmap,self.score_max, ap_dis,lof_dis,update = self.local_track(image_ori)

    def get_first_state(self):
        return self.score_map,self.score_pmap, self.score_max

    def metric_init(self, im, init_box):
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
        # print(gt_pos_examples.shape[0])
        while gt_pos_examples.shape[0] < 10:
            gt_iou = gt_iou - 0.05
            gt_pos_examples = pos_generator(init_box[0].astype(np.float32), 20, [gt_iou, 1])
            # print(gt_pos_examples.shape[0])
        with torch.no_grad():
            self.gt_pos_features=self.get_metric_feature(im,gt_pos_examples).cpu().detach().numpy()

        self.clf = lof_fit(self.gt_pos_features, k=5)
        self.lof_thresh = 2.5#2.1
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
        lof_score, _ = lof(box_features.cpu().detach().numpy(), self.clf, k=5, thresh=self.lof_thresh)
        return ap_dist.data.cpu().numpy()[0],lof_score

    def local_init(self, image, init_bbox):
        rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        info = edict()
        info.arch = 'Ocean'
        info.dataset = 'LASOT'
        info.TRT = False
        info.epoch_test = False

        siam_info = edict()
        siam_info.arch = 'Ocean'
        siam_info.dataset = 'LASOT'
        siam_info.online = True
        siam_info.epoch_test = False
        siam_info.TRT = False
        siam_info.align=False
        self.siam_tracker= Ocean(siam_info)
        self.siam_net = models.__dict__['Ocean'](align=siam_info.align, online=siam_info.online)
        self.siam_net = load_pretrain(self.siam_net, "snapshot/OceanOon.model")
        self.siam_net.eval()
        self.siam_net = self.siam_net.cuda()
        self.online_tracker = ONLINE(info)
        for i in range(100):
            self.siam_net.template(torch.rand(1, 3, 127, 127).cuda())
            self.siam_net.track(torch.rand(1, 3, 255, 255).cuda())


        cx, cy, w, h = get_axis_aligned_bbox(np.array(init_bbox))
        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])
        self.state = self.siam_tracker.init(image, target_pos, target_sz, self.siam_net)  # init tracker
        self.online_tracker.init(image, rgb_im, self.siam_net, target_pos, target_sz, True, dataname='LASOT',
                            resume="snapshot/OceanOon.model")
        gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

    def local_track(self, image_ori):
        image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        # self.state = self.siam_tracker.track(self.state, image_ori)
        self.state,update = self.online_tracker.track(image_ori, image, self.siam_tracker, self.state)
        state = cxy_wh_2_rect(self.state['target_pos'], self.state['target_sz'])
        max_score=self.state['max_score']
        score_map=self.state['score']
        score_pmap=self.state['pscore']
        self.last_gt = [state[1], state[0], state[1]+state[3], state[0]+state[2]]
        local_state = np.array(state).reshape((1, 4))
        ap_dis,lof_dis = self.metric_eval(image, local_state)

        return state, score_map,score_pmap,max_score, ap_dis,lof_dis,update

    def tracking(self, image):
        self.i += 1
        local_state1, self.score_map, self.score_pmap,score_max, ap_dis,lof_dis,update = self.local_track(image)
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
                float(height)], self.score_map,self.score_pmap,iou, score_max, ap_dis,lof_dis,update

