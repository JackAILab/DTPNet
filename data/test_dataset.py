import torch.utils.data as udata
import cv2
import os
from utils.utils import SSIM,normalize,batch_PSNR
import numpy as np
import torch
from patchify import patchify

Log_path = "./Logs"

def progress(y_origin):
    b, g, r = cv2.split(y_origin)
    y = cv2.merge([r, g, b])
    
    y = normalize(np.float32(y))
    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    
    high = y.shape[2]//32
    wight = y.shape[3]//32
    y = y[:,:, 0:high*32, 0:wight*32]
    y = torch.Tensor(y)
    return y


class Dataset_Rain200(udata.Dataset):
    def __init__(self, data_style='train'):
        data_path = eval(f'Rain_Rain200H_{data_style}_path')
        super(Dataset_Rain200, self).__init__()
        self.data_path = data_path
        self.target_names =  data_path+'/norain/'
        self.input_names =  data_path+'/rain/'
        self.ls1 = os.listdir(data_path + "rain")
    def __len__(self):
        return len(self.ls1)
    def __getitem__(self, index):
        if ('200' in self.data_path):
            target_names = self.target_names + f'norain-{index+1}.png'
            input_names =  self.input_names + f'norain-{index+1}x2.png'
        elif  ('100' in self.data_path):
            target_names = self.target_names + 'norain-%03d.png'%(index+1)
            input_names = self.input_names + 'rain-%03d.png'%(index+1)
            
        y_origin = cv2.imread(os.path.join(input_names))
        gt = cv2.imread(os.path.join(target_names))
        gt = progress(gt)
        y = progress(y_origin)
        return y, gt

class Dataset_Rain200H(Dataset_Rain200):
    pass 


class Dataset_Rain100(udata.Dataset):
    def __init__(self, data_style='train'):
        data_path = eval(f'Rain_Rain200H_{data_style}_path')
        super(Dataset_Rain100, self).__init__()
        self.data_path = data_path
        self.target_names =  data_path+'/norain/'
        self.input_names =  data_path+'/rain/'
        self.ls1 = os.listdir(data_path + "rain")
    def __len__(self):
        return len(self.ls1)
    def __getitem__(self, index):
        target_names = self.target_names + 'norain-%03d.png'%(index+1)
        input_names = self.input_names + 'rain-%03d.png'%(index+1)
            
        y_origin = cv2.imread(os.path.join(input_names))
        gt = cv2.imread(os.path.join(target_names))
        gt = progress(gt)
        y = progress(y_origin)
        return y, gt

class Dataset_DID(udata.Dataset):
    def __init__(self, data_style):
        super(Dataset_DID, self).__init__()
        data_path = eval(f'Rain_DID_{data_style}_path')
        target_names =  data_path + 'norain/' 
        input_names =  data_path+'rain/'
        ls1 = os.listdir(data_path + "rain")
        self.targets =  [target_names + i for i in ls1]
        self.inputs =  [input_names + i for i in ls1]
    def __len__(self):
            return len(self.targets)
    def __getitem__(self, index):
        target_names = self.targets[index]
        input_names = self.inputs[index]
        y_origin = cv2.imread(input_names)
        gt = cv2.imread(os.path.join(target_names))
        gt = progress(gt).squeeze()
        y = progress(y_origin).squeeze()

        return y, gt
