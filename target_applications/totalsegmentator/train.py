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

from model.SwinUNETR import SwinUNETR
from model.unet3d import UNet3D
from monai.networks.nets import SegResNet
from dataset.dataloader import get_loader
from utils.utils import dice_score, check_data, TEMPLATE, get_key, NUM_CLASS
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


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

    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    # prepare the 3D model  
    # swin unetr pre-trained by us
    if args.model_backbone == "selfswin":
        model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
                        in_channels=1,
                        out_channels=args.num_class,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False
                        )
        store_dict = model.state_dict()
        model_dict = torch.load(args.pretrain, map_location='cpu')["state_dict"]
        amount = 0
        for key in model_dict.keys():
            new_key = '.'.join(key.split('.')[1:])
            if 'out' not in new_key:
                if 'rotation' in new_key:
                    break
                else:
                    store_dict[new_key] = model_dict[key]
                    amount += 1
        print(amount, len(model_dict.keys()))
        model.load_state_dict(store_dict)
        print('Use Self-supervised Swin Unetr pretrained weights')

    # swin unetr released by monai
    if args.model_backbone == 'selfswinunetr':
        model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
                        in_channels=1,
                        out_channels=args.num_class,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False
                        )
        store_dict = model.state_dict()
        model_dict = torch.load(args.pretrain)["state_dict"]
        amount = 0
        for key in model_dict.keys():
            if 'out' not in key:
                store_dict[key] = model_dict[key]
                amount += 1
        print(amount, len(model_dict.keys()))
        model.load_state_dict(store_dict)
        print('Use Self-supervised Swin Unetr pretrained weights by monai')

    # 50,000 pre-trained weights released by monai
    if args.model_backbone == 'ssl':
        model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
                        in_channels=1,
                        out_channels=args.num_class,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False
                        )
        ssl_dict = torch.load(args.pretrain)
        ssl_weights = ssl_dict["model"]

        monai_loadable_state_dict = OrderedDict()
        model_prior_dict = model.state_dict()
        model_update_dict = model_prior_dict

        del ssl_weights["encoder.mask_token"]
        del ssl_weights["encoder.norm.weight"]
        del ssl_weights["encoder.norm.bias"]
        del ssl_weights["out.conv.conv.weight"]
        del ssl_weights["out.conv.conv.bias"]

        for key, value in ssl_weights.items():
            if key[:8] == "encoder.":
                if key[8:19] == "patch_embed":
                    new_key = "swinViT." + key[8:]
                else:
                    new_key = "swinViT." + key[8:18] + key[20:]
                monai_loadable_state_dict[new_key] = value
            else:
                monai_loadable_state_dict[key] = value

        model_update_dict.update(monai_loadable_state_dict)
        model.load_state_dict(model_update_dict, strict=True)
        model_final_loaded_dict = model.state_dict()

        layer_counter = 0
        for k, _v in model_final_loaded_dict.items():
            if k in model_prior_dict:
                layer_counter = layer_counter + 1

                old_wts = model_prior_dict[k]
                new_wts = model_final_loaded_dict[k]

                old_wts = old_wts.to("cpu").numpy()
                new_wts = new_wts.to("cpu").numpy()
                diff = np.mean(np.abs(old_wts, new_wts))
                print("Layer {}, the update difference is: {}".format(k, diff))
                if diff == 0.0:
                    print("Warning: No difference found for layer {}".format(k))
        print("Total updated layers {} / {}".format(layer_counter, len(model_prior_dict)))
        print("50,000 Pretrained Weights Successfully Loaded !")

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
            print(amount, len(store_dict.keys()))
            print('Use SuPreM UNet backbone pretrained weights')
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
            print('Use SuPreM SegResNet backbone pretrained weights')
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
            model.load_state_dict(checkpoint['net'])
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint['net']
            for key in model_dict.keys():
                store_dict['.'.join(key.split('.'))] = model_dict[key]
            model.load_state_dict(store_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler, val_loader, test_loader = get_loader(args)

    best_dice = 0

    if rank == 0:
        writer = SummaryWriter(log_dir='out/' + args.log_name)
        print('Writing Tensorboard logs to ', 'out/' + args.log_name)

    while args.epoch < args.max_epoch:
        if args.dist:
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
                if not os.path.isdir('out/' + args.log_name):
                    os.mkdir('out/' + args.log_name)
                torch.save(checkpoint, 'out/' + args.log_name + '/best_model.pth')
                print('The best model saved at epoch:', args.epoch)
        
        checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
        directory = 'out/' + args.log_name
        if not os.path.isdir(directory):
            os.mkdir(directory)
        torch.save(checkpoint, directory + '/model.pth')

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
    ## model load
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default=None, 
                        help='The path of pretrain model; for unet: ./pretrained_weights/epoch_400.pth')
    parser.add_argument('--trans_encoding', default='rand_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='./pretrained_weights/word_embedding.pth', 
                        help='The path of word embedding')

    parser.add_argument('--max_epoch', default=201, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=20, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')

    parser.add_argument('--dataset_list', nargs='+', default=['total_set'])

    parser.add_argument('--data_root_path', default='...', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')

    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-250, type=float, help='a_min in ScaleIntensityRanged')
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
    parser.add_argument('--map_type', default='organs', help='organs | muscles | cardiac | vertebrae | ribs') 
    parser.add_argument('--num_class', default=18, type=int, help='class num: 18 | 22 | 19 | 25 | 25')
    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')
    parser.add_argument('--datasetkey', nargs='+', default=['01', '02', '03', '04', '05', 
                                            '07', '08', '09', '12', '13', '10_03', 
                                            '10_06', '10_07', '10_08', '10_09', '10_10'],
                                            help='the content for ')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.005, type=float, help='The percentage of cached data in total')
    parser.add_argument('--overlap', default=0.5, type=float, help='overlap for sliding_window_inference')
    parser.add_argument('--dataset_path', default='...', help='dataset path')
    parser.add_argument('--model_backbone', default='unet', help='model backbone, also avaliable for swinunetr')
    parser.add_argument('--percent', default=1081, type=int, help='percent of training data')

    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()
