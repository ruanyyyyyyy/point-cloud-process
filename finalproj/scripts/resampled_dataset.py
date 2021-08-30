import os
import numpy as np
import torch.utils.data as torch_data

class KITTIPCDClassificationDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train', mode='TRAIN'):
        self.data_root = './data/resampled_KITTI'
        filename_labels = 'object_names.txt'
        filename_train = 'train.txt'
        size_validate = 0.20
        filename_test = 'test.txt'
        random_seed = 42

        with open(os.path.join(self.data_root, filename_labels)) as f:
            self.labels = [l.strip() for l in f.readlines()]
        
        self.label2ind = {label: ind for ind, label in enumerate(labels)}
        self.ind2label = {ind: label for ind, label in enumerate(labels)}

        self.__N = 64 # num. of obeservations in point cloud
        
        
        # load training set
        with open(os.path.join(self.data_root, filename_train)) as f:
            self.__train = [e.strip() for e in f.readlines()] # vehicle_001229
        # create validation set:
        
        
        