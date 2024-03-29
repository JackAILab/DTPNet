import torch 
import os
import argparse
import numpy as np
from torchsummary import summary
from thop import profile # JackAdd for complex assesment
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
parser = argparse.ArgumentParser(description="DTPNet_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")  
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs") 
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="./temp", help='path to save models and log files')  
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=3, help='number of recursive stages')
opt = parser.parse_args(args=[])


import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from utils import SSIM, findLastCheckpoint, batch_PSNR
from tqdm import tqdm
from networks import DTPnet
from dataset import Datase_h5f

device = torch.device('cuda')

GPU_NUM = torch.cuda.device_count()
GPU_NUM = 0 #  
if  GPU_NUM>1:
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    ## Initialization
    torch.distributed.init_process_group(backend="nccl" )
    torch.cuda.set_device(rank)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

def main():

    print('Loading dataset ...\n')
    
    dataset_train = Datase_h5f("/data/ProjectData/Derain/Rain200L/train")
    model = DTPnet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu).cuda()
    
    # loader_train = DataLoader(dataset=dataset_train, num_workers=16, batch_size=opt.batch_size, shuffle=True)
    if  GPU_NUM>1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle = (train_sampler is None), sampler=train_sampler, pin_memory=True)
    else:
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle = True,  pin_memory=True)
        
    print("# of training samples: %d\n" % int(len(dataset_train)))
    criterion = SSIM()
    criterion_L2 = nn.MSELoss()

    if  GPU_NUM>1:
        device_ids = [0, 1]
        model = torch.nn.DataParallel(model,device_ids = device_ids)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank,find_unused_parameters=True).cuda()
    # record training
    writer = SummaryWriter(opt.save_path)   

    # load the lastest model  
    # initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    initial_epoch = 0
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        if  GPU_NUM>1:
            model.module.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))
        else:
            # model = DTPnet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu).cuda()
            model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))
            
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,eps=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates
    # start training
    step = 0 
    sum = 0
    for epoch in range(initial_epoch, opt.epochs):
        if  GPU_NUM>1:
            train_sampler.set_epoch(epoch)   
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"]) # learning rate 0.001000
        ## epoch training start
        for i, (input_train, target_train) in enumerate(tqdm(loader_train)):
            # sum += batch_PSNR(input_train,target_train,1.0)
            # print(sum/(i+1))
            # continue
            model.train()   
            # model.zero_grad()
            optimizer.zero_grad()
            
            count, psnr_all= 0,0
            # input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()
            # with autocast():
            out_train, pred1 = model(input_train)

            if i==0:
                # ============== parameter assement ===========================
                # import pdb
                # pdb.set_trace()
                # print(input_train.shape)
                # Parameters
                summary(model, input_size=(3, 96, 96)) # (3, 100, 100)
                #  FLOPs
                flops, params = profile(model, inputs=(input_train, ))
                print(f"FLOPs: {flops}, Parameters: {params}")
                # ============== parameter assement ===========================
            
            pixel_metric = criterion(target_train, out_train) # target_train.size()---torch.Size([2, 3, 100, 100]); out_train.size()---torch.Size([2, 3, 100, 100])
            layer2_loss = criterion_L2(target_train,pred1)
            loss = -pixel_metric  
            loss = -pixel_metric + 0.5*layer2_loss  
            loss.backward() 
            optimizer.step() # 
            
            # training curve
            psnr_all += batch_PSNR(out_train, target_train, 1.)* out_train.shape[0]
            count += out_train.shape[0]
            step += 1
            if step % 1 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_all/count, step)
                Note=open(opt.save_path+'/log.txt','a')
                Note.write("[epoch %d][%d/%d] loss: %.4f, loss2: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
              (epoch+1, i+1, len(loader_train), loss.item(), layer2_loss.item(), pixel_metric.item(), psnr_all/count))
                Note.write('\n')
                # print results
                print("[epoch %d][%d/%d] loss: %.4f, loss2: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), layer2_loss.item(), pixel_metric.item(), psnr_all/count))
            

        if  GPU_NUM>1:
            # if epoch % opt.save_freq == 0 and rank == 0:
            if epoch % opt.save_freq == 0 :
                torch.save(model.module.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))
                torch.save(model.module.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        else:
            if epoch % opt.save_freq == 0:
                torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))
                torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))

        scheduler.step(epoch)   
                    
        if epoch == 300: #           
            from test_PReNet import test
            optimizer.zero_grad() 
            psnr_test_average,pixel_metric_average = test(model)
            # # 
            out_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            im_target = utils.make_grid(target_train.data, nrow=8, normalize=False, scale_each=True)
            im_input = utils.make_grid(input_train.data, nrow=8, normalize=False, scale_each=True)
            im_derain = utils.make_grid(out_train.data, nrow=8, normalize=False, scale_each=True)
            writer.add_image('clean image', im_target, epoch+1)
            writer.add_image('rainy image', im_input, epoch+1)
            writer.add_image('deraining image', im_derain, epoch+1)

            Note=open(opt.save_path+'/log.txt','a')
            Note.write("=========TEST=======[epoch %d] test_loss: %.4f, pixel_metric: %.4f, test_PSNR: %.4f" %
            (epoch+1, loss.item(), pixel_metric_average, psnr_test_average))
            Note.write('\n')

            print("=========TEST=======[epoch %d] test_loss: %.4f, pixel_metric: %.4f, test_PSNR: %.4f" %
            (epoch+1, loss.item(), pixel_metric_average, psnr_test_average))


if __name__ == "__main__":
    main()

