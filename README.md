# MS-AASGCN
## CenterNet3D: An Anchor free Object Detector for Autonomous Driving (Arxiv 2020) [\[paper\]](https://arxiv.org/abs/2007.07214)
Based on the center point, we propose an anchor-free CenterNet3D Network that performs 3D object detection without anchors. 
Our CenterNet3D uses keypoint estimation to find center points and directly regresses 3D bounding boxes. 
Besides, our CenterNet3D is Non-Maximum Suppression free which makes it more efficient and simpler. On the KITTI benchmark, 
our proposed CenterNet3D achieves competitive performance with other one stage anchor-based methods.

## Updates
2023-01-24: MS-AASGCN

# Installation
Our MS-AASGCN is based on [mmaction2](https://github.com/open-mmlab/mmaction2), Please install with the steps bellow for installation.

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
## Citation
If you find this work useful in your research, please consider cite:
```
@misc{wang2020centernet3dan,
    title={CenterNet3D:An Anchor free Object Detector for Autonomous Driving},
    author={Guojun Wang and Bin Tian and Yunfeng Ai and Tong Xu and Long Chen and Dongpu Cao},
    year={2020},
    eprint={2007.07214},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgement
The code is devloped based on mmdetection3d and mmdetecton, some part of codes are borrowed from SECOND and PointRCNN.  
* [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) 
* [mmdetection](https://github.com/open-mmlab/mmdetection) 
* [mmcv](https://github.com/open-mmlab/mmcv)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [PointRCNN](https://github.com/sshaoshuai/PointRCNN)
