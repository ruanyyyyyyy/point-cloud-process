# Installation

## Requirements

All the codes are tested in the following environment:

- Linux (tested on Ubuntu 20.04)
- Python 3.7
- PyTorch 1.8.1
- cuda 11.2

## Install PointRCNN

a. Clone the repository.
```
git clone --recursive https://github.com/.....git
```

If you forget to add the --recursive parameter, just run the following command to clone the Pointnet2.PyTorch submodule.
```
git submodule update --init --recursive
```

b. Install the dependent python libraries like easydict,tqdm, tensorboardX etc.

c. Build and install the pointnet2_lib, iou3d, roipool3d libraries by executing the following command:
```
sh build_and_install.sh
```

# References:
PointRCNN https://github.com/sshaoshuai/PointRCNN

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
cd scripts
python 2_train_and_eval_modelnet.py
```

Visualizae training loss on Tensorboard
```
tensorboard --logdir=runs --bind_all --port=6006
```
After this the traning loss can be monitored inside local browser at http://localhost:46006/

# Credits
https://github.com/AlexGeControl/3D-Point-Cloud-Analytics