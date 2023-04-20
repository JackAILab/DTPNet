import os
import os.path
import numpy as np
import lmdb
np.random.seed(0) 
import h5py
import torch
import cv2
import torch.utils.data as udata
from utils.utils import normalize
from patchify import patchify
import lmdb,pickle
from numpy import random

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride] 
    TotalPatNum = patch.shape[1] * patch.shape[2] 
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32) 

    for i in range(win): 
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum]) 

def progress(y_origin):
    b, g, r = cv2.split(y_origin)
    y = cv2.merge([r, g, b])
    if y.shape[0]>y.shape[1]: 
        y = normalize(np.float32(y)).transpose(2, 0, 1)
    else:
        y = normalize(np.float32(y)).transpose(2, 1, 0)
    high = y.shape[1]//32
    wight = y.shape[2]//32
    y = y[:, 0:high*32, 0:wight*32]
    return y


class Dataset_Rain200L(udata.Dataset):
    def __init__(self, data_path='./ProjectData/Derain/DID-Data'):
        super(Dataset_Rain200L, self).__init__()
        self.data_path = data_path
        self.target_names =  data_path+'/GT/'
        self.input_names =  data_path+'/input/'
        self.num = 4  
        self.train = os.listdir(self.input_names)

    def __len__(self):
        return len(os.listdir(self.input_names))
    
    def __getitem__(self, index):
        train_str = ','.join(map(str, self.train[index:index+1]))
        input_names = self.input_names + train_str
        base_name = train_str[:-5] + '.jpg'
        if 'H.jpg' in train_str:
            target_names = self.target_names + base_name
        elif 'L.jpg' in train_str:
            target_names = self.target_names + base_name
        else:
            target_names = self.target_names + train_str
        y_origin = cv2.imread(os.path.join(input_names))
        gt = cv2.imread(os.path.join(target_names))
        gt = progress(gt)
        y = progress(y_origin)

        gt_patches = Im2Patch(gt,win=96, stride=96)
        y_patches = Im2Patch(y,win=96, stride=96)


        return y_patches[:, :, :, 1],gt_patches[:, :, :, 1]
    
class Datase_h5f(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Datase_h5f, self).__init__()

        self.data_path = data_path

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key]) 
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()
        return torch.Tensor(input), torch.Tensor(target)
    