import os
import numpy as np
import pandas ad pd

from torch.utils.data import Dataset

class KITTIPCDClsDataset(Dataset):
    def __init__(self, root_dir, filename_labels, split='train'):
    '''
    root_dir: './data/resampled_KITTI'
    filename_labels = 'object_names.txt'
    filename_train = 'train.txt'
    filename_test = 'test.txt'
    '''
        self.data_root = root_dir
        filename = split + '.txt'

        with open(os.path.join(self.data_root, filename_labels)) as f:
            self.labels = [l.strip() for l in f.readlines()]
        
        self.label2ind = {label: ind for ind, label in enumerate(labels)}
        self.ind2label = {ind: label for ind, label in enumerate(labels)}

        self.__N = 64 # num. of obeservations in point cloud
        
        
        with open(os.path.join(self.data_root, filename_train)) as f:
            self.__data = [e.strip() for e in f.readlines()] # vehicle_001229

    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, idx):
        label, pc_ind = self.__data[idx].split('_')
        label_ind = self.label2ind[label]
        filename_pc = os.path.join(self.data_rot, label, pc_ind+'.txt')
        df_point_cloud_with_normal = pd.read_csv(
            filename_pc,
            header=None, names=['x', 'y', 'z', 'nx', 'ny', 'nz']
        )

        # format
        xyz = df_point_cloud_with_normal[['x', 'y', 'z']].values.astype(np.float32)
        normals = df_point_cloud_with_normal[['nx', 'ny', 'nz']].values.astype(np.float32)
        
        return xyz, normals, label_ind
    



class KITTIPCDClsDataset_Wrapper(object):
    def __init__(self, root_dir, config):
    '''
    root_dir: './data/resampled_KITTI'
    filename_labels = 'object_names.txt'
    '''
        self.data_root = root_dir
        self.filename_labels = 'object_names.txt'
        
        self.size_validate = 0.20
        self.__N = 64 # num. of obeservations in point cloud
 
        self.bs = config.batch_size
        self.num_workers = config.num_workers

    def get_dataloader(self):
        # load training set
        train_dataset = KITTIPCDClsDataset('./data/resampled_KITTI', self.filename_labels, 'train')

        # create validation set
        dset_size = len(train_dataset)
        dset_indices = list(range(dset_size))
        np.random.shuffle(dset_indices)
        val_split_index = int(np.floor(self.size_validate * dset_size))

        train_idx, val_idx = dset_indices[val_split_index:], dset_indices[:val_split_index]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.bs, sampler=train_sampler, num_workers=self.num_workers, drop_last=True)
        valid_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.bs, sampler=val_sampler, num_workers=self.num_workers, drop_last=True)

        # load test set
        test_dataset = KITTIPCDClsDataset('./data/resampled_KITTI', 'object_names.txt', 'test')
        test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=self.bs, num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader, test_loader

        