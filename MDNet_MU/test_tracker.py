import os
from run_tracker import eval_tracking, p_config
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# p = p_config()
# p.tracker = 'MDNet'
# p.model_dir='fps_new'
# p.save_training_data = False
# p.name = p.tracker + '_'+p.model_dir
# eval_tracking('lasot', p=p, mode='test')


p = p_config()
p.tracker = 'MDNet_MU'
p.save_training_data = False
p.name = p.tracker
eval_tracking('lasot', p=p, mode='test')


