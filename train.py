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
import logging
import random
from torch.optim.lr_scheduler import LambdaLR
import torchvision.models as models

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main():
    global args, best_loss
    args = parser.parse_args()
    best_loss = 1e10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.no_feature:
        backbone_model = models.resnet101(pretrained=True)
        ### freeze all the layers except the lat one
        for k,v in backbone_model.named_parameters():
            if not k.startswith('layer4'):
                v.requires_grad = False
        ### change the last linear output dim to 120
        num_fc_ftr = backbone_model.fc.in_features
        # print("fc_input: ", num_fc_ftr)
        backbone_model.fc = torch.nn.Linear(num_fc_ftr, 120)
        if torch.cuda.device_count() > 1:
            backbone_model = nn.DataParallel(backbone_model, device_ids=args.gpus).cuda()
        else:
            backbone_model.to(device)

    model = EgoViT(N=args.N, d_model=120, d_ff=args.dff, pose_dim=args.pose_dim, h=args.h, dropout=args.dropout)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=args.gpus).cuda()
    else:
        model.to(device)

    path = os.getcwd()
    save_path = os.path.join(path, 'logs', args.exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    ### save hyper parameters
    save_hyperparameter(args)
    ### create log
    logger = loadLogger(args)
    ### load checkpoints if exist
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    ### Yuan Ye
    if args.dataset == 'Yuan':
        train_data = MoCapDataset(dataset_path=args.dataset_path, 
                                config_path=args.config_path, 
                                image_tmpl="{:05d}.png", 
                                image_transform=torchvision.transforms.Compose([
                                            Scale(256),
                                            ToTorchFormatTensor(),
                                            GroupNormalize(
                                                mean=[.485, .456, .406],
                                                std=[.229, .224, .225])
                                            ]), 
                                L=args.L,
                                test_mode=False)

        val_data = MoCapDataset(dataset_path=args.dataset_path, 
                                config_path=args.config_path, 
                                image_tmpl="{:05d}.png", 
                                image_transform=torchvision.transforms.Compose([
                                            Scale(256),
                                            ToTorchFormatTensor(),
                                            GroupNormalize(
                                                mean=[.485, .456, .406],
                                                std=[.229, .224, .225])
                                            ]), 
                                L=args.L,
                                test_mode=False)

    if args.dataset == 'EgoMotion':
        train_data = EgoMotionDataset(dataset_path=args.dataset_path, 
                                    config_path=args.config_path,
                                    no_feature = args.no_feature, 
                                    image_tmpl="{:04d}.jpg", 
                                    image_transform=torchvision.transforms.Compose([
                                                Scale(224),
                                                ToTorchFormatTensor(),
                                                GroupNormalize(
                                                mean=[.485, .456, .406],
                                                std=[.229, .224, .225])
                                            ]), 
                                    L=args.L,
                                    test_mode=False)
        val_data = EgoMotionDataset(dataset_path=args.dataset_path, 
                                    config_path=args.config_path,
                                    no_feature = args.no_feature,  
                                    image_tmpl="{:04d}.jpg", 
                                    image_transform=torchvision.transforms.Compose([
                                            Scale(224),
                                            ToTorchFormatTensor(),
                                            GroupNormalize(
                                                mean=[.485, .456, .406],
                                                std=[.229, .224, .225])
                                            ]), 
                                    L=args.L,
                                    test_mode=False)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, 
                              shuffle=True,num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.workers, pin_memory=True)
    
    if(args.optimizer=='SGD'):
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    if(args.optimizer=='Adam'):
        opt = [120, 10, 4000]
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9
            )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *opt)
        )

    if args.evaluate:
        validate(val_loader, model, 0, args=args)
        return
    
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(" Training epoch: {}".format(epoch+1))
        if(args.optimizer=='SGD'):
            adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, optimizer, lr_scheduler, device, logger=logger, backbone=None, args=args)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            logger.info(" Eval epoch: {}".format(epoch + 1))
            loss1 = validate(val_loader, model, device, 50, logger, None, args)

            # remember best prec@1 and save checkpoint
            is_best = loss1 < best_loss
            best_loss = min(loss1, best_loss)
            if not args.no_feature:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'loss': loss1,
                }, save_path, is_best)
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'transformer_state_dict': model.state_dict(),
                    'backbone_state_dict': backbone_model.state_dict(),
                    'loss': loss1,
                }, save_path, is_best)
        # train(train_loader, model, optimizer, epoch, device)

def train(train_loader, model, optimizer, scheduler, device, batch_num=None, logger=None, backbone=None, args=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if batch_num==None:
        max_iter = len(train_loader)
    else:
        max_iter = batch_num

    model.train()
    if not args.no_feature:
        for i, (motion, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data_time.update(time.time() - end)
            label = label.to(device)
            tgt = label
            # print("tgt shape: ", tgt.shape)
            src = motion.to(device)
            # src shape:(batch,length,feature_dim)
            src_mask = (src.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
            # src_mask shape:(batch,1,length)
            tgt = tgt[:, :-1, :]
            # print("tgt shape: ", tgt.shape)
            # tgt shape:(batch,21,51)
            tgt_mask = (tgt.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
            # tgt_mask shape:(batch,1,length)
            mask_ = torch.tensor(subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data))
            # mask_ shape:(1,length,length)
            tgt_mask = tgt_mask & mask_
            # tgt_mask shape:(batch,length,length)
            output = model(src, tgt, src_mask, tgt_mask)
            # output shape:(batch,length,pose_dim)
            if args.norm == 'L1':
                loss = ComputeLoss_nohead(output[:,:-1,:], label[:,1:-1,:], args.L, order='xyz', norm='L1')
            if args.norm == 'L2':
                loss = ComputeLoss_nohead(output[:,:-1,:], label[:,1:-1,:], args.L, order='xyz', norm='L2')
            if torch.isnan(loss).any():
                print("src: ", src)
                print("tgt: ", tgt)
                print("output: ", output)
                print("index: ", i)
            losses.update(loss.item(), label.shape[0])
            # optimizer.zero_grad()
            loss.backward()
            ### gradient clip: 用来限制过大的梯度
            if args.clip_gradient is not None:
                total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info("lr: {:.5f} \tBatch({:>3}/{:>3}) done. Loss: {:.4f}".format(optimizer.param_groups[0]['lr'], i+1, max_iter, loss.data.item()))

            # if i > 2:
            #     break
            # if i % args.print_freq == 0:
            #     print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            #            epoch, i, len(train_loader), batch_time=batch_time,
            #            data_time=data_time, loss=losses, lr=optimizer.param_groups[-1]['lr'])))
    else:
        for i, (image, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data_time.update(time.time() - end)
            label = label.to(device)
            image = image.to(device)
            feature = backbone(image.reshape(-1, 3, 224, 224))
            # print("feature shape: ", feature.shape)
            # feature shape: (batch*length, 120)
            tgt = label
            # print("tgt shape: ", tgt.shape)
            src = feature.reshape(-1, args.L, 120)
            # src shape:(batch,length,feature_dim)
            src_mask = (src.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
            # src_mask shape:(batch,1,length)
            tgt = tgt[:, :-1, :]
            # print("tgt shape: ", tgt.shape)
            # tgt shape:(batch,21,51)
            tgt_mask = (tgt.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
            # tgt_mask shape:(batch,1,length)
            mask_ = torch.tensor(subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data))
            # mask_ shape:(1,length,length)
            tgt_mask = tgt_mask & mask_
            # tgt_mask shape:(batch,length,length)
            output = model(src, tgt, src_mask, tgt_mask)
            # output shape:(batch,length,pose_dim)
            if args.norm == 'L1':
                loss = ComputeLoss_nohead(output[:,:-1,:], label[:,1:-1,:], args.L, order='xyz', norm='L1')
            if args.norm == 'L2':
                loss = ComputeLoss_nohead(output[:,:-1,:], label[:,1:-1,:], args.L, order='xyz', norm='L2')
           
            losses.update(loss.item(), label.shape[0])
            # optimizer.zero_grad()
            loss.backward()
            ### gradient clip: 用来限制过大的梯度
            if args.clip_gradient is not None:
                total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info("lr: {:.5f} \tBatch({:>3}/{:>3}) done. Loss: {:.4f}".format(optimizer.param_groups[0]['lr'], i+1, max_iter, loss.data.item()))
            

def validate(val_loader, model, device, batch_num=None, logger=None, backbone=None, args=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if batch_num==None:
        max_iter = len(val_loader)
    else:
        max_iter = batch_num

    with torch.no_grad():
        if not args.no_feature:
            for i, (motion, label) in tqdm(enumerate(val_loader), total=max_iter):
                label = label.to(device)
                tgt = label
                src = motion.to(device)
                # src shape:(batch,length,feature_dim)
                src_mask = (src.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
                tgt = tgt[:, :-1, :]
                # src_mask shape:(batch,1,length)
                tgt_mask = (tgt.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
                # tgt_mask shape:(batch,1,length)
                mask_ = torch.tensor(subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data))
                # mask_ shape:(1,length,length)
                tgt_mask = tgt_mask & mask_
                # tgt_mask shape:(batch,length,length)
                output = model(src, tgt, src_mask, tgt_mask)
                # output shape:(batch,length,pose_dim)label = label.to(device)     
                if args.norm == 'L1':
                    loss = ComputeLoss_nohead(output[:,:-1,:], label[:,1:-1,:], args.L, order='xyz', norm='L1')
                if args.norm == 'L2':
                    loss = ComputeLoss_nohead(output[:,:-1,:], label[:,1:-1,:], args.L, order='xyz', norm='L2')
                losses.update(loss.item(), label.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    logger.info(" \tBatch({:>3}/{:>3}) done. Loss:{:.4f}".format(i+1, max_iter, loss.data.item()))
                
                if i > max_iter:
                    break
                # if i % args.print_freq == 0:
                #     print(('Test: [{0}/{1}]\t'
                #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                #            i, len(val_loader), batch_time=batch_time, loss=losses)))
            logger.info('Testing Results: Loss {loss.avg:.5f}'.format(loss=losses))

        else:
            for i, (image, label) in tqdm(enumerate(val_loader), total=max_iter):
                label = label.to(device)
                image = image.to(device)
                feature = backbone(image.reshape(-1, 3, 224, 224))
                # print("feature shape: ", feature.shape)
                # feature shape: (batch*length, 120)
                tgt = label
                # print("tgt shape: ", tgt.shape)
                src = feature.reshape(-1, args.L, 120)
                # src shape:(batch,length,feature_dim)
                src_mask = (src.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
                # src_mask shape:(batch,1,length)
                tgt = tgt[:, :-1, :]
                # print("tgt shape: ", tgt.shape)
                # tgt shape:(batch,21,51)
                tgt_mask = (tgt.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
                # tgt_mask shape:(batch,1,length)
                mask_ = torch.tensor(subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data))
                # mask_ shape:(1,length,length)
                tgt_mask = tgt_mask & mask_
                # tgt_mask shape:(batch,length,length)
                output = model(src, tgt, src_mask, tgt_mask)
                # output shape:(batch,length,pose_dim)
                if args.norm == 'L1':
                    loss = ComputeLoss_nohead(output[:,:-1,:], label[:,1:-1,:], args.L, order='xyz', norm='L1')
                if args.norm == 'L2':
                    loss = ComputeLoss_nohead(output[:,:-1,:], label[:,1:-1,:], args.L, order='xyz', norm='L2')
            
                losses.update(loss.item(), label.shape[0])
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    logger.info(" \tBatch({:>3}/{:>3}) done. Loss:{:.4f}".format(i+1, max_iter, loss.data.item()))
                
                if i > max_iter:
                    break
            logger.info('Testing Results: Loss {loss.avg:.5f}'.format(loss=losses))
    return loss

def save_checkpoint(state, save_path, is_best=True, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, filename))
    file_path = os.path.join(save_path, filename)
    torch.save(state, file_path)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, 'model_best.pth.tar'))
        best_path = os.path.join(save_path, best_name)
        shutil.copyfile(file_path, best_path)

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """
        lr_steps为预先给定的epoch列表
        比如为[10,30,50]
        那么一旦当前epoch数大于10 学习率就衰减为原来的0.1倍
        大于30 再次衰减
        ......
    """
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr * param_group['lr_mult']
    #     param_group['weight_decay'] = decay * param_group['decay_mult']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_hyperparameter(args):
    path = os.getcwd()
    basedir = os.path.join(path, 'logs', args.exp_name)
    f = os.path.join(basedir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

def loadLogger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)

    logger.addHandler(sHandler)
    path = os.getcwd()
    basedir = os.path.join(path, 'logs', args.exp_name)
    
    work_dir = os.path.join(basedir,
                            time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)

    logger.addHandler(fHandler)

    return logger

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

if __name__ == '__main__': 
    setup_seed(42)
    main()