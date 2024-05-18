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

from backbone.Universal_model import Universal_model
from monai.networks.nets import SegResNet
from utils import loss
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

torch.multiprocessing.set_sharing_strategy('file_system')
    
def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE):
    """
    Performs a single training epoch for the segmentation model.

    Args:
        args (argparse.Namespace): Configuration arguments, please change the args to your own settings below.
        train_loader (dataset.dataloader_bdmap): Dataloader providing training data.
        model (model.Universal_model): The segmentation model.
        optimizer (LinearWarmupCosineAnnealingLR): Optimizer to update model parameters.
        loss_seg_DICE (utils.loss): Loss function object (DiceLoss).
        loss_seg_CE (utils.loss): Loss function object (Multi_BCELoss).

    Returns:
        tuple: A tuple containing (average Dice loss, average BCE loss) for the epoch.
    """
    model.train()
    loss_bce_ave = 0
    loss_dice_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, name = batch["image"].to(args.device), batch["label"].float().to(args.device), batch['name']
        # print('x:', x.shape, 'y:', y.shape)
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
    """
    Coordinates model setup, loading, training, and checkpoint saving for a segmentation task.

    Args:
        args (argparse.Namespace): Configuration arguments, please change the args to your own settings below.
    """
    rank = 0
    
    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    if args.backbone == 'segresnet':
        model = SegResNet(
                    blocks_down=[1, 2, 2, 4],
                    blocks_up=[1, 1, 1],
                    init_filters=args.segresnet_init_filters,
                    in_channels=1,
                    out_channels=args.num_class,
                    dropout_prob=0.0,
                    )
    else:
        # Universal model, support for swin_unetr, unet, and other models
        model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                        in_channels=1,
                        out_channels=args.num_class,
                        backbone=args.backbone,
                        encoding=args.trans_encoding
                        )

    # load pre-trained weights
    if args.pretrain:
        model.load_params(torch.load(args.pretrain)["state_dict"])
        print('Use pretrained weights')

    if args.trans_encoding == 'word_embedding':
        if args.backbone != 'segresnet':
            if args.word_embedding:
                word_embedding = torch.load(args.word_embedding)
                model.organ_embedding.data = word_embedding.float()
                print('load word embedding')

    model.to(args.device)
    model.train()

    # initialize for distributed training (if enabled)
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[args.device]) # synchronize in distributed setting
    
    # loss functions, optimizer, scheduler
    loss_seg_DICE = loss.DiceLoss(num_classes=args.num_class).to(args.device)
    loss_seg_CE = loss.Multi_BCELoss(num_classes=args.num_class).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.dist:
            model.load_state_dict(checkpoint['net'])
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint['net']
            for key in model_dict.keys():
                store_dict['.'.join(key.split('.')[1:])] = model_dict[key]
            model.load_state_dict(store_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler = get_loader(args)

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join('out' , args.log_name))

    while args.epoch <= args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        scheduler.step()

        loss_dice, loss_bce = train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE)
        
        if rank == 0:
            writer.add_scalar('train_dice_loss', loss_dice, args.epoch)
            writer.add_scalar('train_bce_loss', loss_bce, args.epoch)
            writer.add_scalar('lr', scheduler.get_lr(), args.epoch)
            
            # saving the whole model, backbone branch, and language branch respectively
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
            
            if args.backbone != 'segresnet':
                checkpoint_backbone = {
                    "net": model.module.backbone.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    "epoch": args.epoch
                }
                checkpoint_language = {
                    "net": {k: v for k, v in model.module.state_dict().items() if k not in model.module.backbone.state_dict()},
                    'optimizer':optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    "epoch": args.epoch
                }
                save_backbone_path = os.path.join(save_dir, args.backbone+'_backbone.pth')
                torch.save(checkpoint_backbone, save_backbone_path)
                print('Backbone branch saved successfully at epoch:', args.epoch)
                
                save_language_path = os.path.join(save_dir, args.backbone+'_language.pth')
                torch.save(checkpoint_language, save_language_path)
                print('Language branch saved successfully at epoch:', args.epoch)

        args.epoch += 1

    dist.destroy_process_group()

def main():
    """
    Parses command line arguments, please change the args to your own settings below.
    Sets up logging, and calls the `process` function for training.
    """ 
    parser = argparse.ArgumentParser()
    
    ## for distributed training
    parser.add_argument('--dist', dest='dist', action="store_true", default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument('--num_workers', default=12, type=int, help='workers numebr for DataLoader')
    
    ## logging
    parser.add_argument('--log_name', default='AbdomenAtlas1.1.unet', help='The path resume from checkpoint')
    
    ## model load
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default=None, 
                        help='The path of pretrain model')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default=None, 
                        help='The path of word embedding, need to change if you change the AbdomenAtlas version, 1.1 by default')
    
    ## hyperparameter
    parser.add_argument("--epoch", default=0)
    parser.add_argument('--max_epoch', default=800, type=int, help='Number of training epoches')
    parser.add_argument('--warmup_epoch', default=20, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    parser.add_argument('--backbone', default='unet', help='model backbone, unet backbone by default')
    parser.add_argument('--segresnet_init_filters', default=16, type=int, help='Number of initial filters in segresnet') 
    
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['AbdomenAtlas1.1'])
    parser.add_argument('--data_root_path', default='...', help='data root path')
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
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='the percentage of cached data in total')
    parser.add_argument('--cache_num', default=3000, type=int, help='the number of cached data')
    parser.add_argument('--num_class', default=25, type=int, help='number of class in AbdomenAtlas, 25 by default for 1.1 version, 9 for 1.0 version')
    parser.add_argument('--dataset_version', default='AbdomenAtlas1.1', help='dataset version for AbdomenAtlas, 1.1 by default')

    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()
