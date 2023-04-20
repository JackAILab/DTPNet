import cv2
import os
import argparse
import numpy as np
import torch
from Net.networks import CTPnet
import time
import tqdm
import torchvision.utils as utils
import data.test_dataset as test_dataset
from utils.utils import SSIM,normalize,batch_PSNR
from torch.utils.data import DataLoader
import yaml
from yaml import Loader
from data.test_dataset import progress

# set yaml path and load yaml config
cfg_path = "./YMAL/DID.yaml"
cfg = yaml.load(open(cfg_path, "r").read(), Loader=Loader)
cfg_name = cfg_path.split('/')[-1].replace(".yaml","")
print("\n"*10+"Now running "+ cfg_name +"\n"*5)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="Test1")
parser.add_argument("--use_GPU", type=bool, default=False, help='use GPU or not')
parser.add_argument("--recurrent_iter", type=int, default=10, help='number of recursive stages')
parser.add_argument("-Log_path", type=str, default='./Logs/', help='number of recursive stages')
opt = parser.parse_args()

# add yaml config into opt
for name in cfg:
    vars(opt)[name] = cfg[name]
opt.Log_path = opt.Log_path + cfg_name

# add dirs and set dir struction
os.system(f"mkdir -p {opt.Log_path}")
os.system(f"mkdir -p {opt.Log_path}/paras")
os.system(f"mkdir -p {opt.Log_path}/results")
os.system(f"touch -p {opt.Log_path}/results.txt")


def test(model,data_name,test_all):
    '''
        model: 
            Net for tested.
        datapath: 
            Dir of the data.
        test_all: bool 
            Test all saved models or latest models paras.
        name_index: int 
            The model index.
    '''

    data_set = eval(f"test_dataloader.Dataset_{data_name}('test')")
    model_path = f"{opt.Log_path}/paras"
    test_loader = DataLoader(dataset=data_set, batch_size=opt.batchsize, shuffle=False, num_workers=4, drop_last=False  )
    model = model.cuda()
    if test_all == True:
        load_names = [ os.path.join(model_path, f'net_epoch{i+1}.pth') for i in range(len(os.listdir(model_path))-1)]
    elif model_path is not None:
        load_names = [os.path.join(model_path, 'net_latest.pth')]
    for name_index, load_name in enumerate(load_names):
        model.load_state_dict(torch.load(load_name))
        print('load_model from ' + load_name)
        model.eval()
        psnr_test ,pixel_metric,count,psnr_max,ssim_max = 0,0,0,0,0
        with torch.no_grad(): 
            if opt.use_GPU:
                torch.cuda.synchronize()
            times = 0
            for rainy, gt in tqdm.tqdm(test_loader):
                rainy, gt = rainy.cuda(), gt.cuda()
                # print(rainy.shape, gt.shape)
                begin = time.time()
                out, _ = model(torch.squeeze(rainy,0))
                endtime = time.time()
                out = torch.clamp(out, 0., 1.)
                criterion = SSIM()
                loss = criterion(out, gt) * out.shape[0]
                pixel_metric += loss
                psnr_cur = batch_PSNR(out,  gt, 1.) * out.shape[0]
                psnr_test += psnr_cur
                if psnr_cur >= psnr_max:
                    psnr_max = psnr_cur
                if loss >= ssim_max:
                    ssim_max = loss
                count += out.shape[0]

            psnst_average = psnr_test / count
            pixel_metric_average = pixel_metric / count
            Note = open(opt.Log_path + "/results.txt",'a')
            Note.write("Epoch %d [Test SSIM is] %0.2f, [Test PSNR is] %0.2f  [Current PSNR]%0.2f [max ]%0.2f maxssim %0.2f==================\n" % (name_index,pixel_metric/ count, psnr_test/ count,psnr_cur,psnr_max,ssim_max))
            print("[Test SS IM is] %0.2f, [Test PSNR is] %0.2f  [Current PSNR]%0.2f [max ]%0.2f maxssim %0.2f==================" % (pixel_metric/ count, psnr_test/ count,psnr_cur,psnr_max,ssim_max))
    return psnst_average.item(),psnr_max.item(),pixel_metric_average.item(),ssim_max.item(),times/count 

if __name__ == "__main__":

    model = CTPnet(recurrent_iter=3, use_GPU=True).cuda()
    res = []
    res += test(model,opt.data_name,False)
    print(res)    
    