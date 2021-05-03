# Installation

## Requirements

All the codes are tested in the following environment:

- Linux (tested on Ubuntu 20.04)
- Python 3.7
- PyTorch 1.8.1

## Install PointRCNN

a. Clone the repository.

`git clone --recursive https://github.com/.....git`

If you forget to add the --recursive parameter, just run the following command to clone the Pointnet2.PyTorch submodule.

`git submodule update --init --recursive`

b. Install the dependent python libraries like easydict,tqdm, tensorboardX etc.

c. Build and install the pointnet2_lib, iou3d, roipool3d libraries by executing the following command:

`sh build_and_install.sh`
