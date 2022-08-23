import os
from run_tracker import eval_tracking, p_config
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# p = p_config()
# p.tracker = 'Eco'
# # p.name = p.tracker
# p.name = p.tracker
# p.save_training_data=True
# eval_tracking('lasot', p=p, mode='train')

# p = p_config()
# p.tracker = 'Eco_MU_ori'
# p.name = p.tracker
# p.save_training_data=False
# eval_tracking('lasot', p=p, mode='test')

p = p_config()
p.tracker = 'Eco_MU'
p.name = p.tracker
p.save_training_data=False
p.lof_thresh=2.5
eval_tracking('lasot', p=p, mode='test)