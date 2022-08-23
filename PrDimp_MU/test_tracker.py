import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings('ignore')

# p = p_config()
# p.tracker = 'PrDimp'
# p.name = p.tracker
# p.save_training_data=False
# eval_tracking('lasot', p=p, mode='test')
# eval_tracking('votlt19', p=p, mode='all')
# eval_tracking('tlp', p=p, mode='all')

# p = p_config()
# p.tracker = 'PrDimp_MU_ori'
# p.name = p.tracker
# p.save_training_data=False
# eval_tracking('lasot', p=p, mode='test')
# eval_tracking('votlt19', p=p, mode='all')
# eval_tracking('tlp', p=p, mode='all')

p = p_config()
p.tracker = 'PrDimp_MU_ori'
p.name = p.tracker+'_29'
p.save_training_data=False
# eval_tracking('got10k', p=p, mode='test')
eval_tracking('lasot', p=p, mode='test')
# eval_tracking('votlt19', p=p, mode='all')
# eval_tracking('tlp', p=p, mode='all')





