import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist, decollate_batch, DistributedSampler
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from model import configs, networkarch
from model.SwinUNETR import SwinUNETR
from model.unet3d import UNet3D
from monai.networks.nets import SegResNet
from dataset.dataloader import get_loader
from utils.utils import dice_score, check_data, TEMPLATE, get_key, NUM_CLASS
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.generate_model_medical_net import generate_model

torch.multiprocessing.set_sharing_strategy('file_system')

dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

def train(args, train_loader, model, optimizer):
    model.train()
    loss_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, name = batch["image"].to(args.device), batch["label"].float().to(args.device), batch['name']
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (loss=%2.5f)" % (
                args.epoch, step, len(train_loader), loss.item())
        )
        loss_ave += loss.item()
        torch.cuda.empty_cache()
    print('Epoch=%d: ave_loss=%2.5f' % (args.epoch, loss_ave/len(epoch_iterator)))
    
    return loss_ave/len(epoch_iterator)


def validation(model, ValLoader, args):
    model.eval()
    dice_stat = np.zeros((2, args.num_class))
    post_label = AsDiscrete(to_onehot=args.num_class)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.num_class)
    for index, batch in enumerate(tqdm(ValLoader)):
        image, val_labels, name = batch["image"].to(args.device), batch["label"].to(args.device), batch["name"]
        with torch.no_grad():
            val_outputs = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=args.overlap, mode='gaussian')
        val_labels_list = decollate_batch(val_labels)
        val_labels_convert = [
            post_label(val_label_tensor) for val_label_tensor in val_labels_list
        ]
        val_outputs_list = decollate_batch(val_outputs)
        val_output_convert = [
            post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
        ]        
        dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    dice_array = dice_metric.aggregate()
    mean_dice_val = dice_metric.aggregate().mean().item()
    dice_metric.reset()
    return mean_dice_val

def process(args):
    rank = 0

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()% torch.cuda.device_count()
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)
    # SuPreM swinunetr backbone
    if args.model_backbone == 'swintransformer' and args.pretraining_method_name=="smit":
        config = configs.get_SMIT_small_128_bias_True()
        model = networkarch.SMIT_3D_Seg(config,
                                        out_channels=args.num_class,pretrain=args.pretrain)
        args.roi_x,args.roi_y,args.roi_z=128,128,128
        print(f'Use {args.pretraining_method_name} swintransformer backbone pretrained weights')
    # SuPreM swinunetr backbone
    if args.model_backbone == 'swinunetr':
        model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
                        in_channels=1,
                        out_channels=args.num_class,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False
                        )
        # Load pre-trained weights
        store_dict = model.state_dict()
        model_dict = torch.load(args.pretrain)['net']
        amount = 0
        for key in model_dict.keys():
            new_key = '.'.join(key.split('.')[1:])
            if 'backbone' in new_key:
                n_key = '.'.join(new_key.split('.')[1:])
                if n_key in store_dict.keys():
                    store_dict[n_key] = model_dict[key]
                    amount += 1
        print(amount, len(model_dict.keys()))
        model.load_state_dict(store_dict)
        print(f'Use {args.pretraining_method_name} SwinUnetr backbone pretrained weights')
    
    # SuPreM unet backbone
    if args.model_backbone == 'unet':
        model = UNet3D(n_class=args.num_class)
        if args.pretrain is not None:
            if args.pretraining_method_name=="suprem":
                model_dict = torch.load(args.pretrain)['net']
                store_dict = model.state_dict()
                amount = 0
                for key in model_dict.keys():
                    new_key = '.'.join(key.split('.')[2:])
                    if new_key in store_dict.keys():
                        store_dict[new_key] = model_dict[key]   
                        amount += 1

                model.load_state_dict(store_dict)
                print(amount, len(store_dict.keys()))
            elif args.pretraining_method_name=="genesis":
                model_dict = torch.load(args.pretrain)['state_dict']
                store_dict = model.state_dict()
                amount=0
                for key in model_dict.keys():
                    if 'out_tr' not in key:
                        new_key = '.'.join(key.split('.')[1:])
                        if new_key in store_dict.keys():
                            store_dict[new_key] = model_dict[key]   
                            amount += 1
                model.load_state_dict(store_dict)
                print(amount, len(store_dict.keys()))
            elif args.pretraining_method_name=="med3d":
                class Opt:
                    def __init__(self, model_depth, input_W, input_H, input_D, resnet_shortcut, no_cuda, n_seg_classes,pretrain_path):
                        self.model_depth = model_depth
                        self.input_W = input_W
                        self.input_H = input_H
                        self.input_D = input_D
                        self.resnet_shortcut = resnet_shortcut
                        self.no_cuda = no_cuda
                        self.n_seg_classes = n_seg_classes
                        self.pretrain_path = pretrain_path
                        self.new_layer_names = ['conv_seg']
                # Example usage:
                opt = Opt(model_depth=101, input_W=args.roi_x, input_H=args.roi_y, input_D=args.roi_z, resnet_shortcut='B', no_cuda=False, n_seg_classes=args.num_class,pretrain_path=args.pretrain)
                model,para=generate_model(opt)
            elif args.pretraining_method_name=="dodnet":
                model = MOTS_model(args, norm_cfg='IN', activation_cfg='relu', num_classes=args.num_classes,
                           weight_std=False, deep_supervision=False, res_depth=args.res_depth, dyn_head_dep_wid=dyn_head_dep_wid)
            else:
                print(f"unkown: {args.pretraining_method_name}")
            print(f'Use {args.pretraining_method_name} UNet backbone pretrained weights')
        else:
            print('This is training from scratch')

    # SuPreM segresnet backbone
    if args.model_backbone == 'segresnet':
        model = SegResNet(
                    blocks_down=[1, 2, 2, 4],
                    blocks_up=[1, 1, 1],
                    init_filters=16,
                    in_channels=1,
                    out_channels=args.num_class,
                    dropout_prob=0.0,
                    )
        if args.pretrain is not None:
            model_dict = torch.load(args.pretrain)['net']
            store_dict = model.state_dict()
            amount = 0
            for key in model_dict.keys():
                new_key = '.'.join(key.split('.')[1:])
                if new_key in store_dict.keys() and 'conv_final.2.conv' not in new_key:
                    store_dict[new_key] = model_dict[key]   
                    amount += 1
            model.load_state_dict(store_dict)
            print(amount, len(store_dict.keys()))
            print(f'Use {args.pretraining_method_name} SegResNet backbone pretrained weights')
        else:
            print('This is SegResNet training from scratch')
            
    model.to(args.device)
    model.train()

    model = DistributedDataParallel(model, device_ids=[args.device])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler, val_loader, test_loader = get_loader(args)

    best_dice = 0

    if rank == 0:
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        writer = SummaryWriter(log_dir=os.path.join('out',args.log_name))
        print('Writing Tensorboard logs to ', os.path.join('out',args.log_name))

    while args.epoch < args.max_epoch:
        dist.barrier()
        train_sampler.set_epoch(args.epoch)
        scheduler.step()
        loss = train(args, train_loader, model, optimizer)
        if rank == 0:
            writer.add_scalar('train_loss', loss, args.epoch)
            writer.add_scalar('lr', scheduler.get_lr(), args.epoch)

        if (args.epoch % args.store_num == 0):
            mean_dice = validation(model, val_loader, args)
            if mean_dice > best_dice:
                best_dice = mean_dice
                checkpoint = {
                    "net": model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    "epoch": args.epoch
                }
                if not os.path.isdir(os.path.join('checkpoints',args.log_name)):
                    os.mkdir(os.path.join('checkpoints',args.log_name))
                torch.save(checkpoint, os.path.join('checkpoints',args.log_name,'best_model.pth') )
                print('The best model saved at epoch:', args.epoch)
        
        checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
        directory = os.path.join('checkpoints',args.log_name)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        torch.save(checkpoint, os.path.join(directory,'model.pth') )

        args.epoch += 1

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='unet_organ', help='The path resume from checkpoint')
    parser.add_argument('--pretraining_method_name', default='suprem', help='name of the model')
    ## model load
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default=None, 
                        help='The path of pretrain model; for unet: ./pretrained_weights/epoch_400.pth')
    parser.add_argument('--max_epoch', default=201, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=20, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')

    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')

    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-200, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=800, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')
    parser.add_argument('--map_type', default='cas', help='depend on your target task') 
    parser.add_argument('--num_class', default=2, type=int, help='count of target task class + 1(background) ')
    parser.add_argument('--overlap', default=0.5, type=float, help='overlap for sliding_window_inference')
    parser.add_argument('--dataset_path', default='...', help='dataset path')
    parser.add_argument('--model_backbone', default='unet', help='model backbone:unet|swintransformer|segresnet|swinunetr')
    parser.add_argument('--fold', default=1, type=int, help='data fold')

    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()
