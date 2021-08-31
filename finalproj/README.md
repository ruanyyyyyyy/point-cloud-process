# Installation

## Requirements

All the codes are tested in the following environment:

- Linux (tested on Ubuntu 20.04)
- Python 3.7
- PyTorch 1.8.1
- cuda 11.2

# KITTI dataset
Refer to the paper about recording platform, sensor setup, structure of the provided zip-files, development kit, object coordinates and transformations among different coordinates. The paper provides detailed explanations about those information.
http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

# Object Classification Network
PointNet++ with cross entropy loss is used as object classification network

Use pointnet++ implementation in https://github.com/sshaoshuai/Pointnet2.PyTorch 

## Install Pointnet++ library

Install the library by running the following command:

```
cd pointnet2_lib
cd pointnet
python setup.py install
```
Then, you can use pointnet2 by importing `pointnet2_msg_cls` or `pointnet2_msg_sem.py` in `pointnet2_lib/tools`

An example:
```
from pointnet2_lib.tools import pointnet2_msg_cls 

MODEL = pointnet2_msg_cls # import network module
model = MODEL.get_model(input_channels=0)
```

Note that `pointnet2_msg_sem.py` is in the original github repo. I modified its forward function in it and got `pointnet2_msg_cls` for classification task.

## Training object classification network

Run the following commands to train the network on the resampled dataset:

```bash
python pointnet2/train.py --ckpt_save_interval 2 --epochs 20
```

After 20 epochs the network can achieve 0.92 accuracy on validation dataset

Resume from previous ckpt
```
python pointnet2/train.py --ckpt_save_interval 2 --epochs 40 --resume true --ckpt output/default/ckpt/checkpoint_epoch_20.pth --lr 0.00016
```

Visualizae training loss on Tensorboard
```
tensorboard --logdir=runs
```
After this the traning loss can be monitored inside local browser

Generate confusion matrix and the classification report from sklearn
```
python pointnet2/train.py --mode test --ckpt output/default/ckpt/checkpoint_epoch_20.pth
```
Confusion matrix png is stored in `./output/confusion_matrix.png`. The classification report is stored in `./output/default/test/log.txt`