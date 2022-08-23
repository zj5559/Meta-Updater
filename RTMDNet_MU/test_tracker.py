import os
from run_tracker import eval_tracking, p_config
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test RTMD+MU
p = p_config()
p.tracker = 'RTMD_MU'#RTMD_MU
p.save_training_data = False

p.lof_thresh=2.5
p.name = p.tracker
eval_tracking('lasot', p=p, mode='test')

