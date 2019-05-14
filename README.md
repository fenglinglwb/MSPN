# Rethinking on Multi-Stage Networks for Human Pose Estimation
----

## Introduction
This is a pytorch realization of MSPN proposed in [ Rethinking on Multi-Stage Networks for Human Pose Estimation ][1]. In this work, we design an effective network MSPN to perform human pose estimation problem.

Existing pose estimation approaches fall into two categories: single-stage and multi-stage methods. While multistage methods are seemingly more suited for the task, their performance in current practice is not as good as singlestage methods. This work studies this issue. We argue that the current multi-stage methods’ unsatisfactory performance comes from the insufficiency in various design choices. We propose several improvements, including the single-stage module design, cross stage feature aggregation, and coarse-tofine supervision. 

The resulting method establishes the new state-of-the-art on both MS COCO and MPII Human Pose dataset, justifying the effectiveness of a multi-stage architecture.

| Model | Dataset | Input Size | mAP | PCKh@0.5 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 1-stg MSPN | COCO val | 256x192 | 71.5 | - |
| 2-stg MSPN | COCO val | 256x192 | 74.5 | - |
| 3-stg MSPN | COCO val | 256x192 | 75.2 | - |
| 4-stg MSPN | COCO val | 256x192 | 75.9 | - |
| 4-stg MSPN | COCO test-dev | 384x288 | 76.1 | - |
| 4-stg MSPN<sup>*</sup> | COCO test-dev | 384x288 | 77.1 | - |
| 4-stg MSPN<sup>+*</sup> | COCO test-dev | 384x288 | 78.1 | - |
| 4-stg MSPN<sup>+*</sup> | COCO test-challenge | 384x288 | 76.4 | - |
| 4-stg MSPN | MPII val | 256x256 | - | 91.2 |
| 4-stg MSPN<sup>#</sup> | MPII test | 256x256 | - | 92.6 |

#### Note
* \* means using external data
* \+ means using model ensemble
* \# means using multi-shift test

## Quick Start

### Installation

1. Install Pytorch refering to [official web][2].

2. Clone this repo, and config MSPN_HOME in '/etc/profile' or '~/.bashrc', e.g.
 ```
 export MSPN_HOME='root of your cloned repo'
 export PYTHONPATH=$PYTHONPATH:$MSPN_HOME
 ```

3. Install requirements.
 ```
 pip3 install -r requirements.txt
 ```

4. Install COCOAPI
 ```
 git clone https://github.com/cocodataset/cocoapi.git $MSPN_HOME/lib/COCOAP
 cd $MSPN_HOME/lib/COCOAPI/PythonAPI
 python3 setup.py install --user
 ```
 
### Dataset

##### COCO
Download images from [COCO website][3] and put train2017/val2017 splits to **$MSPN_HOME/dataset/COCO/images/**

##### MPII
Download images from [MPII website][4] and put all images into **$MSPN_HOME/dataset/MPII/images/**

### Train
Go to specified experiment repository, e.g.
```
cd $MSPN_HOME/exps/mspn.2xstg.coco
```
and run
```
python config.py -log
python -m torch.distributed.launch --nproc_per_node=gpu_num train.py
```
the ***gpu_num*** is the number of gpus you want to use.

### Test
```
python -m torch.distributed.launch --nproc_per_node=gpu_num test.py -i iter_num
```
the ***gpu_num*** is the number of gpus you want to use, and ***iter_num*** is the specified iteration model.

## Citation
Please cite
```
@article{li2019rethinking,
  title={Rethinking on Multi-Stage Networks for Human Pose Estimation},
  author={Li, Wenbo and Wang, Zhicheng and Yin, Binyi and Peng, Qixiang and Du, Yuming and Xiao, Tianzi and Yu, Gang and Lu, Hongtao and Wei, Yichen and Sun, Jian},
  journal={arXiv preprint arXiv:1901.00148},
  year={2019}
}
```

[1]: https://arxiv.org/abs/1901.00148
[2]: https://pytorch.org/
[3]: http://cocodataset.org/#download
[4]: http://human-pose.mpi-inf.mpg.de/


