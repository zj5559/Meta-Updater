import os
import numpy as np
import cv2
import time
import sys
main_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if main_path not in sys.path:
    sys.path.append(main_path)
    sys.path.append(os.path.join(main_path, 'utils'))
from PrDimp_MU.Prdimp_MU import PrDimp_MU_Tracker
from PrDimp_MU.Prdimp import PrDiMP_Tracker
from PrDimp_MU.Prdimp_MU_ori import PrDimp_MU_ori_Tracker
from local_path import uav_dir,lasot_dir, tlp_dir, otb_dir, votlt19_dir, votlt18_dir,got10k_dir,tn_dir,nfs_dir
from tracking_utils import Region
import json


class p_config(object):
    tracker = 'PrDimp_MU'
    name = None
    model_dir = '1'
    save_results = True
    save_training_data = False
    visualization = False
    lof_thresh=2.5


class VOTLT_Results_Saver(object):
    def __init__(self, save_path, video, t):
        result_path = os.path.join(save_path, 'longterm')
        if not os.path.exists(os.path.join(result_path, video)):
            os.makedirs(os.path.join(result_path, video))
        self.g_region = open(os.path.join(result_path, video, video + '_001.txt'), 'w')
        self.g_region.writelines('1\n')
        self.g_conf = open(os.path.join(result_path, video, video + '_001_confidence.value'), 'w')
        self.g_conf.writelines('\n')
        self.g_time = open(os.path.join(result_path, video, video + '_time.txt'), 'w')
        self.g_time.writelines([str(t)+'\n'])

    def record(self, conf, region, t):
        self.g_conf.writelines(["%f" % conf + '\n'])
        self.g_region.writelines(["%.4f" % float(region[0]) + ',' + "%.4f" % float(
            region[1]) + ',' + "%.4f" % float(region[2]) + ',' + "%.4f" % float(region[3]) + '\n'])
        self.g_time.writelines([str(t)+'\n'])


def get_seq_list(Dataset, mode=None, classes=None):
    if Dataset == "votlt18":
        data_dir = votlt18_dir
    elif Dataset == 'otb':
        data_dir = otb_dir
    elif Dataset == 'uav123':
        data_dir = uav_dir
    elif Dataset == 'got10k':
        data_dir = os.path.join(got10k_dir,mode)
    elif Dataset == 'trackingnet':
        data_dir = tn_dir
    elif Dataset=='nfs':
        data_dir=nfs_dir
    elif Dataset == "votlt19":
        data_dir = votlt19_dir
    elif Dataset == "tlp":
        data_dir = tlp_dir
    elif Dataset == "lasot":
        data_dir = os.path.join(lasot_dir, classes)
    elif Dataset == 'demo':
        data_dir = '../demo_sequences'

    if Dataset in ['votlt18', 'otb', "votlt19", "tlp", 'demo', 'lasot', 'got10k']:
        sequence_list = os.listdir(data_dir)
        sequence_list.sort()
        # sequence_list = [title for title in sequence_list if not title.endswith("txt")]
        sequence_list = [title for title in sequence_list if os.path.isdir(os.path.join(data_dir, title))]
        testing_set_dir = '../utils/testing_set.txt'
        testing_set = list(np.loadtxt(testing_set_dir, dtype=str))
        if mode == 'test' and Dataset == 'lasot':
            print('test data')
            sequence_list = [vid for vid in sequence_list if vid in testing_set]
        elif mode == 'train' and Dataset == 'lasot':
            print('train data')
            sequence_list = [vid for vid in sequence_list if vid not in testing_set]
    elif Dataset == 'uav123':
        with open(os.path.join(data_dir, 'data_seq/UAV123/UAV123.json'), 'r') as f:
            meta_data = json.load(f)
        sequence_list = list(meta_data.keys())
    elif Dataset=='nfs':
        with open(os.path.join(data_dir,'NFS.json'),'r') as f:
            meta_data=json.load(f)
        sequence_list=list(meta_data.keys())
    elif Dataset == 'trackingnet':
        sequence_list = os.listdir(os.path.join(data_dir, 'frames'))
        sequence_list.sort()
        sequence_list = [title for title in sequence_list if os.path.isdir(os.path.join(data_dir, title))]
    # sequence_list.reverse()
    return sequence_list, data_dir


def get_groundtruth(Dataset, data_dir, video):
    if Dataset == "votlt18" or Dataset == "votlt19" or Dataset == "demo":
        sequence_dir = data_dir + '/' + video + '/color/'
        gt_dir = data_dir + '/' + video + '/groundtruth.txt'
    elif Dataset == "otb":
        sequence_dir = data_dir + '/' + video + '/img/'
        gt_dir = data_dir + '/' + video + '/groundtruth_rect.txt'
    elif Dataset == "uav123":
        sequence_dir=os.path.join(data_dir,'data_seq/UAV123')+'/'
        gt_dir=os.path.join(data_dir,'anno/UAV123',video+'.txt')
    elif Dataset == "got10k":
        sequence_dir = os.path.join(data_dir,video)+'/'
        gt_dir = os.path.join(data_dir,video,'groundtruth.txt')
    elif Dataset == "trackingnet":
        sequence_dir = os.path.join(data_dir,'frames',video)+'/'
        gt_dir = os.path.join(data_dir,'anno',video+'.txt')
    elif Dataset == "lasot":
        sequence_dir = data_dir + '/' + video + '/img/'
        gt_dir = data_dir + '/' + video + '/groundtruth.txt'
    elif Dataset == "tlp":
        sequence_dir = data_dir + '/' + video + '/img/'
        gt_dir = data_dir + '/' + video + '/groundtruth_rect.txt'
    if Dataset!='nfs':
        try:
            groundtruth = np.loadtxt(gt_dir, delimiter=',')
        except:
            groundtruth = np.loadtxt(gt_dir)
    if Dataset == 'tlp':
        groundtruth = groundtruth[:, 1:5]
    elif Dataset in ['got10k','trackingnet']:
        if groundtruth.shape[0]==4:
            groundtruth=groundtruth[np.newaxis,:]
    if Dataset=='nfs':
        with open(os.path.join(data_dir,'NFS.json'),'r') as f:
            meta_data=json.load(f)
        sequence_dir=os.path.join(data_dir,meta_data[video]['img_names'][0][:-9])
        groundtruth=np.array(meta_data[video]['gt_rect'])
    return sequence_dir, groundtruth


def run_seq_list(Dataset, p, sequence_list, data_dir,mode):
    if Dataset=='uav123':
        with open(os.path.join(data_dir,'data_seq/UAV123/UAV123.json'),'r') as f:
            meta_data=json.load(f)
    m_shape = 19
    base_save_path = os.path.join('./results', p.name, Dataset)
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)
        if p.save_results and not os.path.exists(os.path.join(base_save_path, 'eval_results')) and \
                not (Dataset == 'got10k' and mode=='test'):
            os.makedirs(os.path.join(base_save_path, 'eval_results'))
        if p.save_training_data and not os.path.exists(os.path.join(base_save_path, 'train_data')):
            os.makedirs(os.path.join(base_save_path, 'train_data'))

    for seq_id, video in enumerate(sequence_list):
        if video=='GOT-10k_Train_004419':
            continue
        sequence_dir, groundtruth = get_groundtruth(Dataset, data_dir, video)

        if p.save_training_data:
            result_save_path = os.path.join(base_save_path, 'train_data', video + '.txt')
            if os.path.exists(result_save_path):
                continue
        if p.save_results:
            if Dataset == 'got10k' and mode=='test':
                if os.path.exists(os.path.join(base_save_path, video)) and \
                        os.path.exists(os.path.join(base_save_path, video, video + '_time.txt')):
                    continue
            else:
                result_save_path = os.path.join(base_save_path, 'eval_results', video + '.txt')
                if os.path.exists(result_save_path):
                    continue

        if Dataset == 'uav123':
            image_list = meta_data[video]['img_names']
        elif Dataset=='nfs':
            with open(os.path.join(data_dir, 'NFS.json'), 'r') as f:
                meta_data = json.load(f)
            image_list = meta_data[video]['img_names']
        else:
            image_list = os.listdir(sequence_dir)
            image_list.sort()
            image_list = [im for im in image_list if im.endswith("jpg") or im.endswith("jpeg")]
        if Dataset == 'otb':
            if video == 'Diving':
                image_list = image_list[:215]
            elif video == 'David':
                image_list = image_list[299:]
            elif video == 'Freeman3':
                image_list = image_list[:460]
            elif video == 'Football1':
                image_list = image_list[:74]
            elif video == 'Freeman4':
                image_list = image_list[:283]
            elif video == 'Board':
                image_list = image_list[:697]
        region = Region(groundtruth[0, 0], groundtruth[0, 1], groundtruth[0, 2], groundtruth[0, 3])
        if Dataset != 'nfs':
            image_dir = sequence_dir + image_list[0]
        else:
            image_dir = os.path.join(data_dir, image_list[0])
        image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
        h = image.shape[0]
        w = image.shape[1]
        region1 = groundtruth[0]
        box = np.array([region1[0] / w, region1[1] / h, (region1[0] + region1[2]) / w, (region1[1] + region1[3]) / h])
        tic = time.time()
        if p.tracker == 'PrDimp_MU':
            tracker = PrDimp_MU_Tracker(image, region, p=p, groundtruth=groundtruth)
        elif p.tracker == 'PrDimp':
            tracker = PrDiMP_Tracker(image, region, p=p, groundtruth=groundtruth)
        elif p.tracker=='PrDimp_MU_ori':
            tracker=PrDimp_MU_ori_Tracker(image, region, p=p, groundtruth=groundtruth)
        else:
            ValueError()
        score_map, score_max = tracker.get_first_state()
        t = time.time() - tic
        time_results = []
        time_results.append(t)
        if p.save_results and Dataset in ['votlt18', 'votlt19']:
            results_saver = VOTLT_Results_Saver(base_save_path, video, t)
        num_frames = len(image_list)
        all_map = np.zeros((num_frames, m_shape, m_shape))
        all_map[0] = cv2.resize(score_map, (m_shape, m_shape))
        bBoxes_results = np.zeros((num_frames, 4))
        bBoxes_results[0, :] = region1
        bBoxes_train = np.zeros((num_frames, 9))
        bBoxes_train[0, :] = [box[0], box[1], box[2], box[3], 0, 1, score_max, 0,0]

        for im_id in range(1, len(image_list)):
            tic = time.time()
            if Dataset != 'nfs':
                imagefile = sequence_dir + image_list[im_id]
            else:
                imagefile = os.path.join(data_dir, image_list[im_id])
            image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
            region, score_map, iou, score_max, dis,lof_dis = tracker.tracking(image)
            print("%d: " % seq_id + video + ": %d /" % im_id + "%d" % len(image_list) + " iou:%f" % iou + " dis:%f" % dis)

            t = time.time() - tic
            time_results.append(t)
            if p.save_results and Dataset in ['votlt18', 'votlt19']:
                results_saver.record(conf=score_max, region=region, t=t)
            all_map[im_id] = cv2.resize(score_map, (m_shape, m_shape))

            box = np.array(
                [region[0] / w, region[1] / h, (region[0] + region[2]) / w, (region[1] + region[3]) / h])
            bBoxes_train[im_id, :] = [box[0], box[1], box[2], box[3], im_id, iou, score_max, dis,lof_dis]
            bBoxes_results[im_id, :] = region
        if p.save_training_data:
            np.savetxt(os.path.join(base_save_path, 'train_data', video + '.txt'), bBoxes_train,
                       fmt="%.8f,%.8f,%.8f,%.8f,%d,%.8f,%.8f,%.8f,%.8f")
            np.save(os.path.join(base_save_path, 'train_data', video + '_map'), all_map)
        if p.save_results:
            if Dataset == 'got10k' and mode=='test':
                if not os.path.exists(os.path.join(base_save_path, video)):
                    os.mkdir(os.path.join(base_save_path, video))
                np.savetxt(os.path.join(base_save_path, video, video + '_001.txt'), bBoxes_results,
                           fmt="%.8f,%.8f,%.8f,%.8f")
                np.savetxt(os.path.join(base_save_path, video, video + '_time.txt'), np.array(time_results),
                           fmt="%.8f")
            else:
                np.savetxt(os.path.join(base_save_path, 'eval_results', video + '.txt'), bBoxes_results,
                           fmt="%.8f,%.8f,%.8f,%.8f")


def eval_tracking(Dataset, p, mode=None):
    if Dataset == 'lasot':
        classes = os.listdir(lasot_dir)
        classes.sort()
        # classes.reverse()
        for c in classes:
            if not os.path.isdir(os.path.join(lasot_dir,c)):
                continue
            sequence_list, data_dir = get_seq_list(Dataset, mode=mode, classes=c)
            run_seq_list(Dataset, p, sequence_list, data_dir,mode)
    elif Dataset in ['votlt18', 'votlt19', 'tlp', 'otb', 'demo','got10k','trackingnet','uav123','nfs']:
        sequence_list, data_dir = get_seq_list(Dataset,mode=mode)
        run_seq_list(Dataset, p, sequence_list, data_dir,mode)
    else:
        print('Warning: Unknown dataset.')
