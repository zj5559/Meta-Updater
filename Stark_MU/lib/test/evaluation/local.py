from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/zj/tracking/SR/code/Stark-main/data/got10k_lmdb'
    settings.got10k_path = '/media/zj/Samsung_T5/Datasets/GOT-10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/home/zj/tracking/SR/code/Stark-main/data/lasot_lmdb'
    settings.lasot_path = '/media/zj/Samsung_T5/Datasets/LaSOT/images'
    settings.network_path = '/home/zj/tracking/SR/code/Stark-main/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = '/home/zj/tracking/LTMU_Expansion/LTMU-master/Stark_MU'
    settings.result_plot_path = '/home/zj/tracking/SR/code/Stark-main/test/result_plots'
    settings.results_path = '/home/zj/tracking/SR/code/Stark-main/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/zj/tracking/LTMU_Expansion/LTMU-master/Stark_MU'
    settings.segmentation_path = '/home/zj/tracking/SR/code/Stark-main/test/segmentation_results'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/zj/4T/Dataset/TrackingNet'
    settings.uav_path = '/media/zj/Samsung_T5/Datasets/Dataset_UAV123/UAV123'
    settings.vot_path = '/home/zj/tracking/SR/code/Stark-main/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

