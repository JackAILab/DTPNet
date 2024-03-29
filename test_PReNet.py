import cv2
import os
import argparse
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.autograd import Variable
from networks import CTPnet
import tqdm
from utils import SSIM,normalize,batch_PSNR
from PIL import Image

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="/home/huangjh/Project/DeRain/JackCode/PreNet_Demo_copy/logs/PreNet_test/", help='path to model and log files')  # 使用自己训练得到的模型
parser.add_argument("--data_path", type=str, default="/home/huangjh/Data/ProjectData/RainData/TraditionalData/test/Rain100L/rainy/", help='path to training data')
parser.add_argument("--save_path", type=str, default="/home/huangjh/Project/DeRain/JackCode/PreNet_Demo_copy/logs/Result/", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=False, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

save_path = "/data/ProjectData/Derain/Rain200H/TrainedModel/mixDTPNet/Logs/derainedIMG/"


# output = output.cpu().detach().numpy().squeeze().transpose(1, 2, 0) # torch.Size([1, 3, 480, 320])
# input = input.squeeze().transpose((1, 2, 0)) # (481, 321, 3)
# gt = gt.cpu().detach().numpy().squeeze().transpose(1, 2, 0) # torch.Size([1, 3, 480, 320])

def save_images(output, input, gt, img_name):
    
    output = output.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    output = (output * 255).astype('uint8')
    output_image = Image.fromarray(output)

    
    input = input.squeeze().transpose(1, 2, 0)  # (481, 321, 3)
    input_image = Image.fromarray(input)
    input_image = input_image.resize((320, 480))  # 调整大小为 (320, 480)

    
    gt = gt.squeeze().transpose(1, 2, 0)  # (3, 480, 320)
    gt = (gt * 255).astype('uint8')
    gt_image = Image.fromarray(gt)

   
    images = [output_image, input_image, gt_image]

    
    combined_image = Image.new('RGB', (3 * 320, 480))
    for i, img in enumerate(images):
        combined_image.paste(img, (i * 320, 0))

    
    file_name = f"{img_name}.jpg"

    
    combined_image.save(file_name)
    print(f"Image Saved at: {file_name}")

def progress(y_origin):
    b, g, r = cv2.split(y_origin)
    y = cv2.merge([r, g, b])
    y = normalize(np.float32(y))
    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    high = y.shape[2]//32
    wight = y.shape[3]//32
    y = y[:,:, 0:high*32, 0:wight*32]
    y = Variable(torch.Tensor(y))
    return y

# data_path = '/data/ProjectData/Derain/Rain100L/rainy'
# data_path = '/data/ProjectData/Derain/Rain200L/test/rain'
data_path = '/data/ProjectData/Derain/Rain200H/test/rain'
log_path = '/data/ProjectData/Derain/Rain200H/TrainedModel/mixDTPNet/Logs'
# log_path = "/data/ProjectData/Derain/Rain200L/TrainedModel/mixDTPNet/Logs/200L-MSEtrick"
# log_path = "/data/ProjectData/Derain/Rain200L/TrainedModel/mixDTPNet/Logs/200L-SSIMtrick"

patch = 1

def test(model):
    model.zero_grad() 
    model = CTPnet(use_GPU=True)
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(log_path, 'net_latest.pth')),strict=False)
    model.eval()
    psnr_test ,pixel_metric,count,psnr_max,ssim_max = 0,0,0,0,0
    for img_name in tqdm.tqdm(os.listdir(data_path)):
        # if is_image(img_name):
            y_origin = cv2.imread(os.path.join(data_path, img_name))
            y = progress(y_origin).cuda()
            with torch.no_grad(): 
                if opt.use_GPU:
                    torch.cuda.synchronize()
                out, _ = model(y)
                # gt_path = os.path.join(data_path.replace('rainy','norain'), 'no'+img_name) #  Rain100H
                gt_path = os.path.join(data_path[:-4]+"norain", img_name.replace('x2','')) #  Rain200L
                gt = progress(cv2.imread(gt_path)).cuda()[:,:,:,:]  

                # import pdb
                # pdb.set_trace()
                # save_images(out, y_origin, gt, img_name)
                                
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

                print("[Test SSIM is] %0.2f, [Test PSNR is] %0.2f  [Current PSNR]%0.2f [max ]%0.2f maxssim %0.2f==================" % (pixel_metric/ count, psnr_test/ count,psnr_cur,psnr_max,ssim_max))

    print("[Test SSIM is] %f, [Test PSNR is] %f ==================" % (pixel_metric/ count, psnr_test/ count))
    
    return pixel_metric/ count,psnr_test/ count

if __name__ == "__main__":
    model = CTPnet(recurrent_iter=3, use_GPU=True).cuda()
    test(model)

