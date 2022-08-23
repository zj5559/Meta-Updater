import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings
warnings.filterwarnings('ignore')

p = p_config()
p.tracker = 'Ocean_MU'
p.name = p.tracker
p.save_training_data=False
p.lof_thresh=2.5
eval_tracking('lasot', p=p, mode='test')







