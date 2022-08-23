import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


p = p_config()
p.tracker = 'Stark_MU'
p.save_training_data = False
p.update_interval=[200]
p.lof_thresh=2.5
p.name = p.tracker
eval_tracking('lasot', p=p, mode='test')





