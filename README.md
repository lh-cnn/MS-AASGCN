# MS-AASGCN
## Skeleton-based multi-stream adaptive-attentional sub-graph convolution network for action recognition [\[paper\]]()

# Introduction
Accurate and fast 3D object detection from point clouds is a key task in autonomous driving. Existing one-stage 3D object detection methods can achieve real-time performance, however, they are dominated by anchor-based detectors which are inefficient and require additional post-processing. In this paper, we eliminate anchors and model an object as a single point the center point of its bounding box. Based on the center point, we propose an anchor-free CenterNet3D Network that performs 3D object detection without anchors. Our CenterNet3D uses keypoint estimation to find center points and directly regresses 3D bounding boxes. However, because inherent sparsity of point clouds, 3D object center points are likely to be in empty space which makes it difficult to estimate accurate boundary. To solve this issue, we propose an auxiliary corner attention module to enforce the CNN backbone to pay more attention to object boundaries which is effective to obtain more accurate bounding boxes. Besides, our CenterNet3D is Non-Maximum Suppression free which makes it more efficient and simpler. On the KITTI benchmark, our proposed CenterNet3D achieves competitive performance with other one stage anchor-based methods which show the efficacy of our proposed center point representation.  

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
# Train
To train the CenterNet3D, run the following command:
```
cd CenterNet3d
python tools/train.py ./configs/centernet3d.py
```
# Test
To evaluate the model, run the following command:
```
cd CenterNet3d
python tools/test.py ./configs/centernet3d.py ./work_dirs/centernet3d/epoch_25.pth
```
