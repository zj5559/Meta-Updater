import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings('ignore')

# p = p_config()
# p.tracker = 'ATOM'
# p.name = p.tracker
# p.save_training_data=True
# eval_tracking('lasot', p=p, mode='train')

p = p_config()
p.tracker = 'ATOM_MU'
p.name = p.tracker
p.save_training_data=False
p.lof_thresh=2.5
eval_tracking('lasot', p=p, mode='test')
# eval_tracking('tlp', p=p, mode='test')
# eval_tracking('uav123', p=p, mode='test')
# eval_tracking('got10k', p=p, mode='test')

# p = p_config()
# p.tracker = 'ATOM_MU_ori'
# p.name = p.tracker
# p.save_training_data=False
# eval_tracking('got10k', p=p, mode='test')









