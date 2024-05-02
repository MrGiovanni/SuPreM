import torch
import numpy as np
from tqdm import tqdm
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from dataset.dataloader_bdmap import get_loader

from tensorboardX import SummaryWriter

from model.unet3d import UNet
from model.SwinUNETR import SwinUNETR
from utils import loss
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

torch.multiprocessing.set_sharing_strategy('file_system')
    
def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE):
    model.train()
    loss_bce_ave = 0
    loss_dice_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, name = batch["image"].to(args.device), batch["label"].float().to(args.device), batch['name']
        logit_map = model(x)
        term_seg_Dice = loss_seg_DICE.forward(logit_map, y)
        term_seg_BCE = loss_seg_CE.forward(logit_map, y)
        loss = term_seg_BCE + term_seg_Dice
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item())
        )
        loss_bce_ave += term_seg_BCE.item()
        loss_dice_ave += term_seg_Dice.item()
        torch.cuda.empty_cache()
    print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)))
    
    return loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)

def process(args):
    rank = 0

    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    # unet backbone by default 
    if args.backbone == 'unet':        
        model = UNet(
                     spatial_dims=3,
                     in_channels=1,
                     out_channels=args.num_class,
                     channels=(64, 128, 256, 512),
                     strides=(2, 2, 2, 2),
                    )
    if args.backbone == 'swinunetr':
        model = SwinUNETR(
                          img_size=(args.roi_x, args.roi_y, args.roi_z),
                          in_channels=1,
                          out_channels=args.num_class,
                          feature_size=48,
                          drop_rate=0.0,
                          attn_drop_rate=0.0,
                          dropout_path_rate=0.0,
                          use_checkpoint=False,
                         )

    model.to(args.device)
    model.train()

    if args.dist:
        model = DistributedDataParallel(model, device_ids=[args.device])
    loss_seg_DICE = loss.DiceLoss(num_classes=args.num_class).to(args.device)
    loss_seg_CE = loss.Multi_BCELoss(num_classes=args.num_class).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler = get_loader(args)

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join('out' , args.log_name))

    while args.epoch < args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        scheduler.step()

        loss_dice, loss_bce = train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE)
        if rank == 0:
            writer.add_scalar('train_dice_loss', loss_dice, args.epoch)
            writer.add_scalar('train_bce_loss', loss_bce, args.epoch)
            writer.add_scalar('lr', scheduler.get_lr(), args.epoch)
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
            save_dir = os.path.join('out', args.log_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, args.backbone+'.pth')
            torch.save(checkpoint, save_path)
            print('Model saved successfully at epoch:', args.epoch)

        args.epoch += 1

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    
    ## for distributed training
    parser.add_argument('--dist', dest='dist', action="store_true", default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument('--num_workers', default=12, type=int, help='workers numebr for DataLoader')
    
    ## logging
    parser.add_argument('--log_name', default='AbdomenAtlas1.0.unet', help='The path resume from checkpoint')
    
    ## hyperparameter
    parser.add_argument("--epoch", default=0)
    parser.add_argument('--max_epoch', default=800, type=int, help='Number of training epoches')
    parser.add_argument('--warmup_epoch', default=20, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    parser.add_argument('--backbone', default='unet', help='model backbone, unet backbone by default') 
    
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['AbdomenAtlas1.0'])
    parser.add_argument('--data_root_path', default='/data2/wenxuan/AbdomenAtlas1.0', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=2, type=int, help='sample number in each ct')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='the percentage of cached data in total')
    parser.add_argument('--cache_num', default=3000, type=int, help='the number of cached data')
    parser.add_argument('--num_class', default=9, type=int, help='number of class')

    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()
