# MS-AASGCN
## Skeleton-based multi-stream adaptive-attentional sub-graph convolution network for action recognition [\[paper\]]()

# Introduction
Graph convolutional networks have achieved remarkable performance with skeleton-based actin recognition methods. However, there is potential correlation between different parts of the human body. Many studies have ignored that different actions are the result of the interaction of different human body parts and that operating on the whole graph provides inadequate information to characterize the action category. In this study, to pay more attention to this problem and further improve the accuracy of action recognition models, sub-graphs based on the depth-first tree traversal order were used to represent the importance and correlation characteristics of joint parts and bone parts. In addition, beyond the physical structure of the body, joint and bone motion information was also introduced to represent changes in human body parts with movement. To make this method work better, an adaptive-attentional mechanism was also introduced to learn unique topology autonomously for each sample and channel domain. Multi-stream adaptive-attentional sub-graph convolution network was thus proposed for action recognition.

## Updates
2023-01-24: MS-AASGCN

# Installation
Our MS-AASGCN is based on [PYSKL toolbox](https://github.com/kennymckormick/pyskl), Please install with the steps bellow for installation.
```
conda create -n ms-aasgcn python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate ms-aasgcn
pip3 install openmim
mim install mmcv-full
mim install mmdet
mim install mmpose
git clone https://github.com/lh-cnn/MS-AASGCN.git
cd ms-aasgcn
pip install -r requirments
pip3 install -e .
```

## Prepare Skeleton Datasets

- [x] NTURGB+D (CVPR 2016): [NTU RGB+D: A large scale dataset for 3D human activity analysis](https://openaccess.thecvf.com/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf)
- [x] NTURGB+D 120 (TPAMI 2019): [Ntu rgb+ d 120: A large-scale benchmark for 3d human activity understanding](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8713892)
- NTURGB+D [2D Skeleton]: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl
- NTURGB+D [3D Skeleton]: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl
# Train
To train one stream, run the following command:
```
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
eg: bash tools/dist_train.sh configs/aasgcn/aasgcn_ntu60_xsub_3dkp/j.py 1 --validate --test-last --test-best
```
# Test
To evaluate one stream, run the following command:
```
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
eg: bash tools/dist_test.sh configs/aasgcn/aasgcn_ntu60_xsub_3dkp/j.py checkpoints/SOME_CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```
