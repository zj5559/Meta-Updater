class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/zj/tracking/SR/code/Stark-main'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/zj/tracking/SR/code/Stark-main/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/zj/tracking/SR/code/Stark-main/pretrained_networks'
        self.lasot_dir = '/media/zj/Samsung_T5/Datasets/LaSOT/images'
        self.got10k_dir = '/media/zj/Samsung_T5/Datasets/GOT-10k'
        self.lasot_lmdb_dir = '/home/zj/tracking/SR/code/Stark-main/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/zj/tracking/SR/code/Stark-main/data/got10k_lmdb'
        self.trackingnet_dir = '/media/zj/4T/Dataset/TrackingNet'
        self.trackingnet_lmdb_dir = '/home/zj/tracking/SR/code/Stark-main/data/trackingnet_lmdb'
        self.coco_dir = '/home/zj/tracking/SR/code/Stark-main/data/coco'
        self.coco_lmdb_dir = '/home/zj/tracking/SR/code/Stark-main/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/zj/tracking/SR/code/Stark-main/data/vid'
        self.imagenet_lmdb_dir = '/home/zj/tracking/SR/code/Stark-main/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
