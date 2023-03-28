import torch
from torch.utils.data import DataLoader
from ego_pose.data_process import MoCapDataset, EgoMotionDataset
from ego_pose.transforms import *
from ego_pose.transformer import *
from ego_pose.loss import *
import shutil
from opts import parser
import torch.optim
import torch.nn.parallel
from torch.nn.utils import clip_grad_norm
import os
import time
from tqdm import tqdm
from utils.visualize import *
from utils.compute_keypoints import load_keypoints
import argparse
from ego_pose.metrics import *
import logging
import math

def loadLogger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)

    logger.addHandler(sHandler)
    path = os.getcwd()
    basedir = os.path.join(path, 'results', args.exp_name)
    
    work_dir = os.path.join(basedir,
                            time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)

    logger.addHandler(fHandler)

    return logger

def inference(label, motion, Model, length, start_pose, end_pose, device, index):
    label = label.unsqueeze(0).to(device)
    tgt = label
    src = motion.unsqueeze(0).to(device)
    # src shape:(batch,length,feature_dim)
    src_mask = (src.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
    # src_mask shape:(batch,1,length)
    tgt_mask = (tgt.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
    # tgt_mask shape:(batch,1,length)
    mask_ = torch.tensor(subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data))
    # mask_ shape:(1,length,length)
    tgt_mask = tgt_mask & mask_

    # tgt_mask shape:(batch,length,length)
    # output = model(src, tgt, src_mask, tgt_mask)
    # output shape:(batch,length,pose_dim)label = label.to(device)
    memory = Model.model.encode(src, src_mask)
    # memory shape: (batch, length, feature_dim)
    ys = start_pose.unsqueeze(1).to(device)
    
    for k in range(length - 1):
        out = Model.model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        pose = Model.model.generator(out[:, -1]).unsqueeze(0)
        # pose shape: (1, pose_dim)->(1, 1, pose_dim)
        ys = torch.cat([ys, pose], dim=1)
    # ys shape: (1, length, pose_dim)
    # label = torch.cat((label[..., 0:9], label[..., 12:21], label[..., 24:]), dim=-1)
    ys = ys.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0)[1:-1].cpu().detach().numpy()
    image_name = "Skeleton"+'_'+str(index)+'.png'
    image_path = os.path.join(save_path, image_name)
    label_name = "Label"+'_'+str(index)+'.png'
    label_path = os.path.join(save_path, label_name)
    print("label shape: ", label.shape)
    DrawSkeleton(label[19,:], head1=None, head2=None, image_name=label_path, dataset='EgoMotion')
    DrawSkeleton(ys[19,:], head1=None, head2=None, image_name=image_path, dataset='EgoMotion')
    # PlotLPose2D(label, length, image_name=label_path, gt_flag=True)
    # PlotLPose2D(ys, length, image_name=image_path, gt_flag=False)
    # for j in range(ys.shape[1]):
    #     image_name = 'Skeleton'+'_'+str(index)+'_'+str(j)+'.jpg'
    #     image_path = os.path.join(save_path, image_name)
    #     label_name = 'Label'+'_'+str(index)+'_'+str(j)+'.jpg'
    #     label_path = os.path.join(save_path, label_name)
    #     keypoint = ys[:, j, :]
    #     # head1 = ys[:, j, 0:3]
    #     # head2 = ys[:, j, 3:6]
    #     label_ = label[:, j, :]
    #     DrawSkeleton(label_.squeeze(0)[:], head1=None, head2=None, image_name=label_path, dataset='EgoMotion')
    #     DrawSkeleton(keypoint[0], head1=None, head2=None, image_name=image_path, dataset='EgoMotion')

def get_args_parser():
    parser = argparse.ArgumentParser('Ego pose Estimation', add_help=False)
    parser.add_argument('--dataset_path', type=str, help='path to your dataset')
    parser.add_argument('--config_path', type=str, help='path to your config')
    parser.add_argument('--dataset', type=str, choices=['EgoMotion', 'Yuan'])
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--pose_dim', type=int, default=51, help='dimension of joints location')
    parser.add_argument('--N', type=int, default=6, help='number of transformer sublayers')
    parser.add_argument('--L', type=int, default=20, help='length of input')
    parser.add_argument('--h', type=int, default=5, help='num of head')
    parser.add_argument('--dff', type=int, default=720, help='num of hidden neurons in SA')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    return parser

index = 20240
parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
exp_name = args.exp_name
length = args.L
path = os.getcwd()
save_path = os.path.join(path, 'results', 'evaluation', exp_name)
logger = loadLogger(args)
if not os.path.exists(save_path):
    os.makedirs(save_path) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Model = EgoViT(N=args.N, d_model=120, d_ff=args.dff, pose_dim=args.pose_dim, h=args.h, dropout=args.dropout)
dataset_path = args.dataset_path
config_path = args.config_path
### load checkpoints if exist

resume = args.resume
checkpoint = torch.load(resume)
# Model.load_state_dict(checkpoint['state_dict'])
# model = nn.DataParallel(model, device_ids=[0,1]).cuda()
# 多GPU
Model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(resume)['state_dict'].items()})   
Model = Model.to(device)

if args.dataset == 'Yuan':
    val_data = MoCapDataset(dataset_path=dataset_path, 
                                config_path=config_path, 
                                image_tmpl="{:05d}.png", 
                                image_transform=torchvision.transforms.Compose([
                                            Scale(256),
                                            ToTorchFormatTensor(),
                                            GroupNormalize(
                                                mean=[.485, .456, .406],
                                                std=[.229, .224, .225])
                                            ]), 
                                L=length,
                                test_mode=True)
    start_pose_path = "/home/liumin/litianyi/workspace/data/datasets/keypoints/0213_take_01_worldpos.csv"
    end_pose_path = "/home/liumin/litianyi/workspace/data/datasets/keypoints/0213_take_10_worldpos.csv"
if args.dataset == 'EgoMotion':
    val_data = EgoMotionDataset(dataset_path=dataset_path, 
                                config_path=config_path, 
                                image_tmpl="{:04d}.jpg", 
                                no_feature=False,
                                image_transform=torchvision.transforms.Compose([
                                            Scale(256),
                                            ToTorchFormatTensor(),
                                            GroupNormalize(
                                                mean=[.485, .456, .406],
                                                std=[.229, .224, .225])
                                            ]), 
                                L=length,
                                test_mode=True)
    start_pose_path = "/home/liumin/litianyi/workspace/data/EgoMotion/keypoints/02_01_worldpos.csv"
    end_pose_path = "/home/liumin/litianyi/workspace/data/EgoMotion/keypoints/143_19_worldpos.csv"
val_loader = DataLoader(dataset=val_data, batch_size=1, 
                        shuffle=True, num_workers=8, pin_memory=True)
Model.eval()
# start_pose_path = "/home/liumin/litianyi/workspace/data/datasets/keypoints/0213_take_01_worldpos.csv"
# end_pose_path = "/home/liumin/litianyi/workspace/data/datasets/keypoints/1205_take_15_worldpos.csv"
# start_pose = load_keypoints(start_pose_path, 0, 1)
# # shape: (1,51)
# end_pose = load_keypoints(end_pose_path, 1000, 1)

################# 

start_pose = val_data.load_keypoints(start_pose_path, 0, 1)
# # shape: (1,48)
end_pose = val_data.load_keypoints(end_pose_path, 50, 1)
# ### 不使用DataLoader而是直接根据索引读取数据
motion = val_data[index][0]
print("motion shape: ", motion.shape)
label = val_data[index][1]
print("label shape: ", label.shape)
# label_name = "Label"+'_'+str(index)+'.png'
# label_path = os.path.join(save_path, label_name)
# DrawSkeleton(label[21,:], head1=None, head2=None, image_name=label_path, dataset='EgoMotion')
inference(label, motion, Model, length, start_pose, end_pose, device, index)

# mpjpe = 0.
# Empjpe = 0.
# acc =0.
# count = 0
# data = []
# for i, (motion, label) in tqdm(enumerate(val_loader), total=len(val_loader)):
#     label = label.to(device)
#     tgt = label
#     src = motion.to(device)
#     # src shape:(batch,length,feature_dim)
#     src_mask = (src.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
#     # src_mask shape:(batch,1,length)
#     tgt_mask = (tgt.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
#     # tgt_mask shape:(batch,1,length)
#     mask_ = torch.tensor(subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data))
#     # mask_ shape:(1,length,length)
#     tgt_mask = tgt_mask & mask_

#     # tgt_mask shape:(batch,length,length)
#     # output = model(src, tgt, src_mask, tgt_mask)
#     # output shape:(batch,length,pose_dim)label = label.to(device)
#     memory = Model.model.encode(src, src_mask)
#     # memory shape: (batch, length, feature_dim)
#     ys = start_pose.unsqueeze(1).to(device)
#     # print("tgt mask: ", tgt_mask[1, 0, :])
#     for k in range(length - 1):
#         out = Model.model.decode(
#             memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
#         )
#         pose = Model.model.generator(out[:, -1]).unsqueeze(0)
#         # pose shape: (1, pose_dim)->(1, 1, pose_dim)
#         ys = torch.cat([ys, pose], dim=1)
#     # ys shape: (1, length, pose_dim)
#     # label = torch.cat((label[..., 0:9], label[..., 12:21], label[..., 24:]), dim=-1)
#     # ys = ys.cpu().detach().numpy()
#     # label = label.cpu().detach().numpy()
#     # print("label shape: ", label.shape)
#     for j in range(1, ys.shape[1]):
#         image_name = 'Skeleton'+'_'+str(i)+'_'+str(j)+'.jpg'
#         image_path = os.path.join(save_path, image_name)
#         label_name = 'Label'+'_'+str(i)+'_'+str(j)+'.jpg'
#         label_path = os.path.join(save_path, label_name)
#         keypoint = ys[:, j, :]
#         # head1 = ys[:, j, 0:3]
#         # head2 = ys[:, j, 3:6]
#         label_ = label[:, j, :]
#     # label_ = label[:, 10, :]
#     # keypoint = ys[:, 11, :]
#     # mpjpe += MPJPE(17, label_, keypoint)
#         DrawSkeleton(label_.squeeze(0)[:], head1=None, head2=None, image_name=label_path, dataset='EgoMotion')
#         DrawSkeleton(keypoint[0], head1=None, head2=None, image_name=image_path, dataset='EgoMotion')

# #     logger.info(" mpjpe: {}".format(mpjpe))
# #     Empjpe += mpjpe
# #     count += 1
# #     data.append(mpjpe)
# #     mpjpe = 0
# #     if i>300:
# #         break
# # Empjpe = Empjpe/count
# # logger.info(" Empjpe: {}".format(Empjpe))
# # Vel = []
# # for i in range(len(data)-1):
# #     vel = abs(data[i+1] - data[i])
# #     Vel.append(vel.data)
# #     vel = 0.
# # Acc = 0.
# # for i in range(len(Vel)-1):
# #     acc = abs(Vel[i+1] - Vel[i])
# #     Acc += acc
# # Acc = Acc / (len(Vel)-1)
# # print("Acc is : ", Acc)