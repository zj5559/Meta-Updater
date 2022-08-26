# Meta-Updater
Official PyTorch implementation of "Robust Online Tracking with Meta-Updater". (IEEE TPAMI)

## Introduction

This work is extended and improved from our preliminary work published in CVPR 2020, entitled “High-Performance Long-Term Tracking with Meta-Updater” (Oral && Best Paper Nomination). We also refer to the sample-level optimization strategy from our another work, published in ACM MM 2020, entitled "Online Filtering Training Samples for Robust Visual Tracking”.

![MU figure](framework.png)

In this work, we propose an upgraded meta-updater (MU), where the appearance information is strengthened in two aspects. First, the local outlier factor is introduced to enrich the representation of appearance information. Second, we redesign the network to strengthen the role of appearance cues in guiding samples’ classification. Furthermore, sample optimization strategies are introduced for particle-based trackers (e.g., MDNet and RTMDNet) to obtain a better performance.

## Experiments
The proposed module can be easily embedded into other online-update trackers to make their online-update more accurately.
To show its generalization ability, we integrate the upgraded meta-updater (MU) and its original version (MU*) into eight different types of short-term trackers with online updates, including correlation filter-based method (ECO), particle-based methods (MDNet and RTMDNet), representative discriminative methods (ATOM, DiMP, and PrDiMP), Siamese-based method (Ocean), and Transformer-based method (Stark). We perform these trackers on four long-term tracking benchmarks (VOT2020LT, OxUvALT, TLP and LaSOT) and two short-term tracking benchmarks (GOT-10k, UAV123) to demonstrate the effectiveness of the proposed meta-updater.

![Experiments](results.png)

## Models and Results
* **Models:**

Please place the models into the corresponding locations of the project according to the folder order in the following link.

[[Baidu Yun](https://pan.baidu.com/s/16Eqpi0AXuI8z55vy9y2iFg)] (a1ca)

[[Google Drive](https://drive.google.com/drive/folders/1YYysWZBeviJ7nBRwy4c0ZTEpnPoHGIkc?usp=sharing)]

* **Results:**

[[Baidu Yun](https://pan.baidu.com/s/1wGcuFnO0CIHzTwXf2kpttQ)] (i1pw)

[[Google Drive](https://drive.google.com/drive/folders/1HlTgpGNbZr0TUJTxJp0axScyoi9rT-He?usp=sharing)]

## Installation

### Requirements
* python 3.7
* CUDA 10.2
* ubuntu 18.04

### Install environment
```
conda create -n MU python=3.7
source activate MU
cd path/to/XXX_MU //e.g. DiMP_MU
pip install -r requirements.txt
```
(Other settings can refer to the official codes of the specific baseline trackers.)

## Testing
1. Edit the path in the file "local_path.py" to your local path.

2. Place the models into the corresponding locations of the project according to the folder order in the above models link.

3. Test the tracker
```
cd XXX_MU
python test_tracker.py
```
If you want to perform the original meta-updater(MU*), you need to change the @model_constructor in the file "ltr/models/tracking/tcNet.py":
```
model = tclstm_fusion() //for the upgraded meta-updater(MU)
model=tclstm() //for the original meta-updater(MU*)
```

4. Evaluate results
```
python evaluate_results.py
```

## Training
1. Run the baseline tracker and record all results(bbox, response map,...) on LaSOT dataset to the training data of meta-updater.You can modify test_tracker.py like this:
```
p = p_config()
p.tracker = 'Dimp'
p.name = p.tracker
p.save_training_data=True
eval_tracking('lasot', p=p, mode='all')
```
2. Edit the file "ltr/tcopt.py", modify tcopts['lstm_train_dir'] and tcopts['train_data_dir'] like this: 
```
tcopts['lstm_train_dir'] = './models/Dimp_MU' //save path of training models
tcopts['train_data_dir'] = '../results/Dimp/lasot/train_data' //dir of training data
```
3. Run ltr/run_training.py to train the upgraded meta-updater.

If you have any questions when install the environment or run the tracker, please contact me (zj982853200@mail.dlut.edu.cn).


## Reference
* **This work is an extention of:**
  * **[LTMU]** "High-Performance Long-Term Tracking with Meta-Updater". CVPR 2020. (Oral && Best Paper Nomination)
  [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dai_High-Performance_Long-Term_Tracking_With_Meta-Updater_CVPR_2020_paper.pdf)]
  [[code](https://github.com/Daikenan/LTMU)]
  * **[MetricNet]** "Online Filtering Training Samples for Robust Visual Tracking". ACM MM 2020.
  [[paper](https://static.aminer.cn/storage/pdf/acm/20/mm/10.1145/3394171.3413930.pdf)]
  [[code](https://github.com/zj5559/MetricNet)]
  
* **Baseline trackers:**
  * **[ECO]** "ECO: Efficient Convolution Operators for Tracking". CVPR 2017.
  [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Danelljan_ECO_Efficient_Convolution_CVPR_2017_paper.pdf)]
  [[code](https://github.com/visionml/pytracking)]
  
  * **[MDNet]** "Learning Multi-Domain Convolutional Neural Networks for Visual Tracking". CVPR 2016.
  [[paper](https://arxiv.org/abs/1510.07945)]
  [[code](https://github.com/hyeonseobnam/py-MDNet)]
  
  * **[RTMDNet]** "Real-Time MDNet". ECCV 2018.
  [[paper](https://arxiv.org/pdf/1808.08834.pdf)]
  [[code](http://cvlab.postech.ac.kr/~chey0313/real_time_mdnet/)]
  
  * **[ATOM]** "ATOM: Accurate Tracking by Overlap Maximization". CVPR 2019.
  [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Danelljan_ATOM_Accurate_Tracking_by_Overlap_Maximization_CVPR_2019_paper.pdf)]
  [[code](https://github.com/visionml/pytracking)]
  
  * **[DiMP]** "Learning Discriminative Model Prediction for Tracking". ICCV 2019.
  [[paper](https://arxiv.org/pdf/1904.07220.pdf)]
  [[code](https://github.com/visionml/pytracking)]
  
  * **[PrDiMP]** "Probabilistic Regression for Visual Tracking". CVPR 2020.
  [[paper](https://arxiv.org/pdf/2003.12565.pdf)]
  [[code](https://github.com/visionml/pytracking)]
  
  * **[Ocean]** "Ocean: Object-aware Anchor-free Tracking". ECCV 2020.
  [[paper](https://arxiv.org/pdf/2006.10721v2.pdf)]
  [[code](https://github.com/researchmm/TracKit)]
  
  * **[Stark]** "Learning Spatio-Temporal Transformer for Visual Tracking". ICCV 2021.
  [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yan_Learning_Spatio-Temporal_Transformer_for_Visual_Tracking_ICCV_2021_paper.pdf)]
  [[code](https://github.com/researchmm/Stark)]
  
## Acknowledgments
The training code is based on the [PyTracking](https://github.com/visionml/pytracking) toolkit. Thanks for PyTracking for providing useful toolkit for SOT.

Thanks for the preliminary works [LTMU](https://github.com/Daikenan/LTMU) and [MetricNet](https://github.com/zj5559/MetricNet).
