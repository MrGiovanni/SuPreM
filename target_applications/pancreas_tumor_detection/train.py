import torch
import numpy as np
from tqdm import tqdm
import os
import argparse
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet

from model.SwinUNETR import SwinUNETR
from model.unet3d import UNet3D

from dataset.dataloader import get_loader
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import nibabel as nib
import random

# Set fixed random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Call this function with the chosen seed
set_seed(317)
print('Set random seed = 317')

torch.multiprocessing.set_sharing_strategy('file_system')

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

def save_nifti(data, dtype, affine, file_path):
    """
    Save a tensor and its affine transformation as a NIfTI file.

    Args:
        data (torch.Tensor): The image tensor to save.
        affine (np.ndarray): Affine transformation matrix associated with the image.
        file_path (str): Path where the NIfTI file will be saved.
    """
    if data.dim() == 5:  # B, C, H, W, D
        data = data.squeeze(0)  # Remove batch dimension, now C, H, W, D

    if data.shape[0] == 1:  # Single channel data
        data = data.squeeze(0)  # Remove channel dimension, now H, W, D
    else:
        raise ValueError("Data has multiple channels, not sure how to handle")
    
    # Ensure the tensor is on CPU and converted to NumPy
    data = data.cpu().numpy()
    # Create a NIfTI image using the data and the affine matrix
    img = nib.Nifti1Image(data.astype(dtype), affine)
    # Save the NIfTI image to file
    nib.save(img, file_path)

def simplify_key(k):
    '''
    Simplify the key by removing common but irrelevant substrings
    Please modify this function according to your needs for your loaded model
    
    Args:
        k (str): the key to simplify
        
    Returns:
        str: the simplified key
    '''
    for prefix in ['module.', 'features.', 'backbone.', 'model.']:
        k = k.replace(prefix, '')
    return k

def train(args, train_loader, model, optimizer):
    model.train()
    loss_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    if args.print_params:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Total number of parameters: ', pytorch_total_params)
    for step, batch in enumerate(epoch_iterator):
        x, y, name = batch["image"].to(args.device), batch["label"].float().to(args.device), batch['name']
        affines = batch['image_meta_dict']['affine']
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if args.epoch == 0:
            for i in range(x.shape[0]):  # Iterate over the batch dimension
                if not os.path.exists(f'visual/{name[i]}/'):
                    os.makedirs(f'visual/{name[i]}/')
                save_nifti(x[i], 
                           np.int16, 
                           affines[i], 
                           f'visual/{name[i]}/x_epoch_{args.epoch}_iter_{step}_img_{i}.nii.gz',
                          )
                save_nifti(y[i], 
                           np.int8, 
                           affines[i], 
                           f'visual/{name[i]}/y_epoch_{args.epoch}_iter_{step}_img_{i}.nii.gz',
                          )
        
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (loss=%2.5f)" % (
                args.epoch, step, len(train_loader), loss.item())
        )
        loss_ave += loss.item()
        torch.cuda.empty_cache()
    print('Epoch=%d: ave_loss=%2.5f' % (args.epoch, loss_ave/len(epoch_iterator)))
    
    return loss_ave/len(epoch_iterator)

def process(args):
    rank = 0

    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    # prepare the 3D model  
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
        model.load_state_dict(store_dict)
        print('Use SuPreM SwinUnetr backbone pretrained weights')
    
    # SuPreM unet backbone
    if args.model_backbone == 'unet':
        model = UNet3D(n_class=args.num_class)
        if args.pretrain is not None:
            model_dict = torch.load(args.pretrain)['net']
            store_dict = model.state_dict()
            amount = 0
            for key in model_dict.keys():
                new_key = '.'.join(key.split('.')[2:])
                if new_key in store_dict.keys():
                    store_dict[new_key] = model_dict[key]   
                    amount += 1

            model.load_state_dict(store_dict)
            print('Use SuPreM U-Net backbone pretrained weights')
        else:
            print('This is training from scratch')

    # SuPreM segresnet backbone
    if args.model_backbone == 'segresnet':
        model = SegResNet(
                    blocks_down=[1, 2, 2, 4],
                    blocks_up=[1, 1, 1],
                    init_filters=args.segresnet_init_filters,
                    in_channels=1,
                    out_channels=args.num_class,
                    dropout_prob=0.0,
                    )
        if args.pretrain is not None:
            model_dict = torch.load(args.pretrain)['net']
            store_dict = model.state_dict()
            simplified_model_dict = {simplify_key(k): v for k, v in model_dict.items()}
            amount = 0
            for key in store_dict.keys():
                if key in simplified_model_dict and 'conv_final.2.conv' not in key:
                    store_dict[key] = simplified_model_dict[key]
                    amount += 1
            assert amount == (len(store_dict.keys())-2), 'the pre-trained model is not loaded successfully'
            print('loading weights', amount, len(store_dict.keys()))
            model.load_state_dict(store_dict)
        else:
            print('This is SegResNet training from scratch')
            
    model.to(args.device)
    model.train()
    
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[args.device])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.dist:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['net'].items():
                name = 'module.' + k  # add 'module.' to each key
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint['net']
            for key in model_dict.keys():
                simplified_key = simplify_key(key)
                store_dict[simplified_key] = model_dict[key]
            model.load_state_dict(store_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler = get_loader(args)

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.log_checkpoint_savepath, args.log_name))
        print('Writing Tensorboard logs to ', os.path.join(args.log_checkpoint_savepath, args.log_name))

    while args.epoch <= args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        scheduler.step()
        loss = train(args, train_loader, model, optimizer)
        if rank == 0:
            writer.add_scalar('train_loss', loss, args.epoch)
            writer.add_scalar('lr', scheduler.get_lr(), args.epoch)

        model_to_save = model.module if hasattr(model, 'module') else model
        if (args.epoch % args.store_num == 0):
            checkpoint = {
                "net": model_to_save.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
                }
            if not os.path.isdir(os.path.join(args.log_checkpoint_savepath, args.log_name)):
                os.mkdir(os.path.join(args.log_checkpoint_savepath, args.log_name))
            torch.save(checkpoint, os.path.join(args.log_checkpoint_savepath, args.log_name, 'checkpoint_epoch_' + str(args.epoch) + '.pth'))
            print('Checkpoint saved at epoch:', args.epoch)

        checkpoint = {
            "net": model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            "epoch": args.epoch
            }
        torch.save(checkpoint, os.path.join(args.log_checkpoint_savepath, args.log_name, 'model.pth'))

        args.epoch += 1

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', dest='dist', action="store_true", default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='unet_organ', help='The path resume from checkpoint')
    parser.add_argument('--log_checkpoint_savepath', default='out', help='The path to save training logs and checkpoints')
    ## model load
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default=None, 
                        help='The path of pretrain model; for unet: ./pretrained_weights/epoch_400.pth')

    parser.add_argument('--max_epoch', default=200, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=50, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=20, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    parser.add_argument('--segresnet_init_filters', default=16, type=int, help='Number of initial filters in segresnet')
    
    parser.add_argument('--dataset_list', nargs='+', default=['JHH'])
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')

    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-100, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=200, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')
    parser.add_argument('--map_type', default='jhh', help='organs | muscles | cardiac | vertebrae | ribs') 
    parser.add_argument('--num_class', default=5, type=int, help='class num: 18 | 22 | 19 | 25 | 25')
    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.005, type=float, help='The percentage of cached data in total')
    parser.add_argument('--overlap', default=0.5, type=float, help='overlap for sliding_window_inference')
    parser.add_argument('--dataset_path', default='...', help='dataset path')
    parser.add_argument('--model_backbone', default='unet', help='model backbone, also avaliable for swinunetr')
    parser.add_argument('--print_params', action='store_true', default=False, help='print model parameter numbers')
    parser.add_argument('--stage', default='train', help='train or test')

    args = parser.parse_args()
    for arg in vars(args):
        print('{}\t{}'.format(arg, getattr(args, arg)))
    
    process(args=args)

if __name__ == "__main__":
    main()
