import os
import os.path
import numpy as np
import lmdb
np.random.seed(0)
import h5py
import torch
import cv2
import torch.utils.data as udata
from utils import normalize
from patchify import patchify
import lmdb,pickle
from numpy import random

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0] # endc 3
    endw = img.shape[1] # endw 512
    endh = img.shape[2] # endh 384
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride] # [:,0:413:100,0:285:100]   (3,2,3)
    TotalPatNum = patch.shape[1] * patch.shape[2] # 5 * 3
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32) # Y为(3,10000,15)

    for i in range(win): #　win 100
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum]) # Y为(3,10000,15)

def progress(y_origin):
    #  rbg 255 add_channel 32
    b, g, r = cv2.split(y_origin)
    y = cv2.merge([r, g, b])
    if y.shape[0]>y.shape[1]: # 防止一个训练集 shape 不一致
        y = normalize(np.float32(y)).transpose(2, 0, 1)
    else:
        y = normalize(np.float32(y)).transpose(2, 1, 0)
    high = y.shape[1]//32
    wight = y.shape[2]//32
    y = y[:, 0:high*32, 0:wight*32]
    # y = torch.Tensor(y)
    return y


class Dataset_Rain200L(udata.Dataset):
    def __init__(self, data_path='/data/ProjectData/Derain/Rain200L/train'):
        super(Dataset_Rain200L, self).__init__()
        self.data_path = data_path
        self.target_names =  data_path+'/norain/'
        self.input_names =  data_path+'/rain/'
        self.num = 3   # 把一张一张大图分为num份

    def __len__(self):
        return len(os.listdir(self.input_names))
    
    def __getitem__(self, index):
        target_names = self.target_names + f'norain-{index+1}.png'
        input_names =  self.input_names + f'norain-{index+1}x2.png'
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
        target = np.array(target_h5f[key]) # 将数据转化为矩阵
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        # target = progress(target.transpose(2, 1, 0))
        # input = progress(input.transpose(2, 1, 0))

        return torch.Tensor(input), torch.Tensor(target)


def prepare_data_RainTraink15(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    # save_target_path = os.path.join(data_path, 'train_target.h5')
    # save_input_path = os.path.join(data_path, 'train_input.h5')
    save_target_path = os.path.join("/data1/huangjiehui_m22/Derain", 'train_target.h5')
    save_input_path = os.path.join("/data1/huangjiehui_m22/Derain", 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(0,200):
        for jj in range(0,42):
            if jj<21:
                target_file = "image_2/000%03d_%02d_norain_2.png" % (i, jj)
            else:
                target_file = "image_3/000%03d_%02d_norain_3.png" % (i, jj-21)
            # target_file_2 = "image_2_3_rain50/000%03d_%02d_rain_2_50.png" % (i, j)
            # target_file_3 = "image_2_3_rain50/000%03d_%02d_rain_3_50.png" % (i, j)
            # print(target_file_2)

            # import pdb /home/jack/Project/Derain/Revise/Neurocomputing/ablationStudy
            # pdb.set_trace()

            if os.path.exists(os.path.join(target_path,target_file)):
                target = cv2.imread(os.path.join(target_path,target_file))
                b, g, r = cv2.split(target)
                target = cv2.merge([r, g, b])

                if jj<21:
                    input_file = "image_2_3_rain50/000%03d_%02d_rain_2_50.jpg" % (i, jj)
                else:
                    input_file = "image_2_3_rain50/000%03d_%02d_rain_3_50.jpg" % (i, jj-21)

                if os.path.exists(os.path.join(input_path,input_file)):

                    input_img = cv2.imread(os.path.join(input_path,input_file))
                    b, g, r = cv2.split(input_img)
                    input_img = cv2.merge([r, g, b])

                    target_img = target
                    target_img = np.float32(normalize(target_img))
                    target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

                    input_img = np.float32(normalize(input_img))
                    input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                    print("input file: %s # shape: %s" % (input_file, input_img.shape))
                    print("target file: %s # samples: %d" % (target_file, target_patches.shape[3]))

                    for n in range(target_patches.shape[3]):
                        target_data = target_patches[:, :, :, n].copy()
                        target_h5f.create_dataset(str(train_num), data=target_data)

                        input_data = input_patches[:, :, :, n].copy()
                        input_h5f.create_dataset(str(train_num), data=input_data)

                        train_num += 1


                    

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


def normalize(data):
    return data / 255.
    # return data


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]  
    endw = img.shape[1]  
    endh = img.shape[2]  
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]  
    TotalPatNum = patch.shape[1] * patch.shape[2] 
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32) 

    # import pdb
    # pdb.set_trace()
    
    for i in range(win):  
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum]) 
