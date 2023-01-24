# MS-AASGCN
## MS-AASGCN [\[paper\]]()

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
The code is devloped based on mmpose2.  
* [PYSKL toolbox](https://github.com/kennymckormick/pyskl)
