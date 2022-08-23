#coding=utf-8
import cv2 as cv
import os
# from region_to_bbox import region_to_bbox
import time
import tensorflow as tf
import yaml, json
import numpy as np
base_path =os.getcwd()
import sys
sys.path.append(os.path.join(base_path, 'implementation'))
sys.path.append(os.path.join(base_path, 'pyMDNet/modules'))
sys.path.append(os.path.join(base_path, 'pyMDNet/tracking'))
# pymdnet
from pyMDNet.modules.model import *
sys.path.insert(0, os.path.join(base_path, 'pyMDNet'))
from pyMDNet.modules.model import MDNet, BCELoss, set_optimizer
from pyMDNet.modules.sample_generator import SampleGenerator
from PIL import Image
from pyMDNet.modules.utils import overlap_ratio
from pyMDNet.tracking.data_prov import RegionExtractor
from pyMDNet.tracking.run_tracker import *
from pyMDNet.modules.judge_metric import *
from pyMDNet.tracking.bbreg import BBRegressor
from pyMDNet.tracking.gen_config import gen_config
from implementation.otbdataset import *
from implementation.uavdataset import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from numpy.random import seed
import torch
from torch.autograd import Variable
# from tensorflow import set_random_seed
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
# print('before opts1')
opts = yaml.safe_load(open(os.path.join(base_path,'pyMDNet/tracking/options_lasot.yaml'),'r'))
# print('after opts1')

class Region:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
def lof_fit(data,k=5,method='l2'):
    clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', metric=method, contamination=0.1)
    clf.fit(data)
    return clf
def lof(predict, clf,k=5, method='l2',thresh=2):
    # calculating LOF
    predict = -clf._score_samples(predict)
    # predict=predict[200:]
    # identifying outliers
    result=predict<=thresh
    return predict,result
def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou
class MDNet_Tracker(object):
    def __init__(self, image, region, imagefile=None, video=None, p=None, groundtruth=None):
        np.random.seed(0)
        torch.manual_seed(0)
        self.i = 0
        self.p = p
        self.video=video
        if groundtruth is not None:
            self.groundtruth = groundtruth
        else:
            self.groundtruth=None
        init_gt1 = [region.x, region.y, region.width, region.height]
        self.init_metricnet(image, init_gt1)

    def init_metricnet(self, image, init_bbox):
        target_bbox = np.array(init_bbox)
        self.last_result = target_bbox
        self.pymodel = MDNet(os.path.join(base_path, 'pyMDNet/models/mdnet_imagenet_vid.pth'))
        if opts['use_gpu']:
            self.pymodel = self.pymodel.cuda()
        self.pymodel.set_learnable_params(opts['ft_layers'])

        # Init criterion and optimizer
        self.criterion = BCELoss()
        init_optimizer = set_optimizer(self.pymodel, opts['lr_init'], opts['lr_mult'])
        self.update_optimizer = set_optimizer(self.pymodel, opts['lr_update'], opts['lr_mult'])

        tic = time.time()
        # metric
        self.metric_model = model_load(opts['metric_model'])
        #warmup
        tmp=np.random.rand(5,3,107,107)
        tmp = torch.Tensor(tmp)
        tmp = (Variable(tmp)).type(torch.FloatTensor).cuda()
        self.metric_model.eval()
        tmp =self.metric_model(tmp)


        self.target_metric_feature = get_target_feature(self.metric_model, target_bbox, np.array(image))
        self.target_metric_feature_all = []
        self.target_metric_feature_all.append(self.target_metric_feature)

        # Draw pos/neg samples
        pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
            target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

        neg_examples = np.concatenate([
            SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
            SampleGenerator('whole', image.size)(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
        # print(neg_examples)
        neg_examples = np.random.permutation(neg_examples)
        #metric
        ii=0
        self.pos_all=np.zeros(pos_examples.shape[0])
        self.pos_all_feature=np.zeros((pos_examples.shape[0],1024))
        while ii<pos_examples.shape[0]:
            with torch.no_grad():
                pos_metric_feature,pos_metric_dist = get_metric_dist_lof(self.metric_model, pos_examples[ii:ii+50], np.array(image),self.target_metric_feature, opts)
            self.pos_all[ii:ii+50]=pos_metric_dist.cpu().detach().numpy()
            self.pos_all_feature[ii:ii+50]=pos_metric_feature.cpu().detach().numpy()
            ii=ii+50
        self.pos_feature_center =torch.from_numpy(np.mean(self.pos_all_feature,axis=0).reshape((1, 1024))).float().cuda()
        self.clf=lof_fit(self.pos_all_feature[0:opts['n_pos_update']],k=opts['pos_k'],method=opts['method'])
        del pos_metric_feature,pos_metric_dist
        torch.cuda.empty_cache()
        opts['pos_thresh'] = self.pos_all.max() * opts['pos_rate']  # 2.5
        opts['metric_similar_thresh'] = self.pos_all.mean() * opts['similar_rate']
        # print('pos_thresh is:', opts['pos_thresh'])
        # print('similar_thresh is:', opts['metric_similar_thresh'])

        # Extract pos/neg features
        pos_feats = forward_samples(self.pymodel, image, pos_examples, opts)
        neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
        self.feat_dim = pos_feats.size(-1)

        # Initial training
        train(self.pymodel, self.criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], opts=opts)
        del init_optimizer, neg_feats
        torch.cuda.empty_cache()

        # Train bbox regressor
        bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'],
                                         opts['aspect_bbreg'])(
            target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
        bbreg_feats = forward_samples(self.pymodel, image, bbreg_examples, opts)
        self.bbreg = BBRegressor(image.size)
        self.bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
        del bbreg_feats
        torch.cuda.empty_cache()
        # Init sample generators
        self.sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
        self.pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
        self.neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

        # Init pos/neg features for update
        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
        neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
        self.pos_feats_all = [pos_feats]
        self.neg_feats_all = [neg_feats]

        samples = self.sample_generator(target_bbox, opts['n_samples'])
        sample_scores = forward_samples(self.pymodel, image, samples, out_layer='fc6', opts=opts)

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        self.first_score = top_scores.data.cpu().numpy()
        self.spf_total = 0
    def get_first_state(self):
        return self.first_score
    def tracking(self, image):
        tic=time.time()
        self.i += 1
        self.image = image

        target_bbox = self.last_result
        samples = self.sample_generator(target_bbox, opts['n_samples'])
        sample_scores = forward_samples(self.pymodel, image, samples, out_layer='fc6', opts=opts)

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > 0 #and top_dist[9]<opts['pos_thresh']
        with torch.no_grad():
            self.target_metric_feature_tmp = get_target_feature(self.metric_model, target_bbox, np.array(image))
            lof_dis,_ = lof(self.target_metric_feature_tmp.cpu().detach().numpy(), self.clf, thresh=opts['pos_thresh_lof'])
            success1, target_dist = judge_success(self.metric_model, self.target_metric_feature_tmp, self.target_metric_feature, opts)
        if success:
            success=success1
        # Expand search area at failure
        if success:
            self.sample_generator.set_trans(opts['trans'])
        else:
            self.sample_generator.expand_trans(opts['trans_limit'])

        self.last_result = target_bbox
        # Bbox regression
        bbreg_bbox = self.pymdnet_bbox_reg(success, samples, top_idx)

        # Save result
        region = bbreg_bbox

        # Data collect
        if success:
            self.collect_samples_metricnet(image)

        # Short term update
        if not success:
            self.pymdnet_short_term_update()

        # Long term update
        elif self.i % opts['long_interval'] == 0:
            self.pymdnet_long_term_update()

        self.spf_total = self.spf_total+time.time() - tic

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
                iou = _compute_iou(self.groundtruth[self.i], region)
        return region,top_scores.data.cpu().numpy(),iou,target_dist[0],lof_dis[0]

    def collect_samples_metricnet(self,image):
        target_bbox = self.last_result
        pos_examples = self.pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
        #metric
        #pos_samples use lof to filter
        with torch.no_grad():
            # pos_examples = judge_metric_lof(self.metric_model, pos_examples, np.array(image), self.target_metric_feature, self.clf_pos,opts)
            pos_features = get_anchor_feature(self.metric_model, np.array(image), pos_examples)  # anchor_box: (1,4) x,y,w,h
            pos_features = pos_features.cpu().detach().numpy()
        # result=lof(self.pos_all_feature[0:opts['n_pos_update']],pos_features,k=opts['pos_k'],method=opts['method'],thresh=opts['pos_thresh_lof'])
        _,result=lof(pos_features,self.clf,thresh=opts['pos_thresh_lof'])
        pos_examples = pos_examples[result]

        if pos_examples.shape[0]>0:
            pos_feats = forward_samples(self.pymodel, self.image, pos_examples, opts)
            with torch.no_grad():
                dist_tmp = get_metric_dist_by_feature(self.metric_model, self.target_metric_feature_all,self.target_metric_feature_tmp, opts)
            idx_tmp = 0
            for idx in range(dist_tmp.shape[0]):
                if dist_tmp[idx] < opts['metric_similar_thresh']:
                    self.target_metric_feature_all.pop(idx - idx_tmp)
                    self.pos_feats_all.pop(idx - idx_tmp)
                    idx_tmp = idx_tmp + 1
            self.pos_feats_all.append(pos_feats)
            self.target_metric_feature_all.append(self.target_metric_feature_tmp)
        if len(self.pos_feats_all) > opts['n_frames_long']:
            del self.pos_feats_all[0]
            del self.target_metric_feature_all[0]

        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
        with torch.no_grad():
            # print(neg_examples)
            result=judge_metric_center(self.metric_model,neg_examples,np.array(image),self.pos_feature_center,opts)
        neg_examples = neg_examples[result]
        if neg_examples.shape[0] > 0:
            neg_feats = forward_samples(self.pymodel, self.image, neg_examples, opts)
            self.neg_feats_all.append(neg_feats)
        if len(self.neg_feats_all) > opts['n_frames_short']:
            del self.neg_feats_all[0]
    def collect_samples_pymdnet(self):
        target_bbox = self.last_result
        pos_examples = self.pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])

        pos_feats = forward_samples(self.pymodel, self.image, pos_examples, opts)
        self.pos_feats_all.append(pos_feats)
        if len(self.pos_feats_all) > opts['n_frames_long']:
            del self.pos_feats_all[0]

        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
        neg_feats = forward_samples(self.pymodel, self.image, neg_examples, opts)
        self.neg_feats_all.append(neg_feats)
        if len(self.neg_feats_all) > opts['n_frames_short']:
            del self.neg_feats_all[0]

    def pymdnet_short_term_update(self):
        # Short term update
        nframes = min(opts['n_frames_short'], len(self.pos_feats_all))
        pos_data = torch.cat(self.pos_feats_all[-nframes:], 0)
        neg_data = torch.cat(self.neg_feats_all, 0)
        train(self.pymodel, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
              opts=opts)
    def pymdnet_long_term_update(self):
        # Short term update
        pos_data = torch.cat(self.pos_feats_all, 0)
        neg_data = torch.cat(self.neg_feats_all, 0)
        train(self.pymodel, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
              opts=opts)
    def metric_bbox_reg(self, success,bbreg_samples):
        target_bbox = self.last_result
        if success:
            bbreg_samples = bbreg_samples[None, :]
            bbreg_feats = forward_samples(self.pymodel, self.image, bbreg_samples, opts)
            bbreg_bbox = self.bbreg.predict(bbreg_feats, bbreg_samples)
        else:
            bbreg_bbox = target_bbox
        return bbreg_bbox

    def pymdnet_bbox_reg(self, success, samples, top_idx):
        target_bbox = self.last_result
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None, :]
            bbreg_feats = forward_samples(self.pymodel, self.image, bbreg_samples, opts)
            bbreg_samples = self.bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox
        return bbreg_bbox
