import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import argparse
import shutil
import nibabel as nib
from monai.inferers import sliding_window_inference
from model.SwinUNETR import SwinUNETR
from monai.networks.nets import SegResNet
from dataset.dataloader import get_loader, taskmap_set
from utils.utils_test import invert_transform


torch.multiprocessing.set_sharing_strategy('file_system')


def validation(model, ValLoader, val_transforms, args):
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model.eval()
    selected_class_map = taskmap_set[args.map_type]
    for index, batch in enumerate(tqdm(ValLoader)):
        image, name = batch["image"].to(args.device), batch["name_img"]
        image_file_path = os.path.join(args.data_root_path,name[0], 'ct.nii.gz')
        case_save_path = os.path.join(save_dir,name[0].split('/')[0])
        
        if not os.path.isdir(case_save_path):
            os.makedirs(case_save_path)
        organ_seg_save_path = os.path.join(save_dir,name[0].split('/')[0],'segmentations')
        
        if args.copy_ct:
            destination_ct = os.path.join(case_save_path,'ct.nii.gz')
            if not os.path.isfile(destination_ct):
                shutil.copy(image_file_path, destination_ct)
                
        name = name.item() if isinstance(name, torch.Tensor) else name  # Convert to Python str if it's a Tensor
        original_affine = nib.load(image_file_path).affine
        with torch.no_grad():
            image = batch["image"].to("cuda")  # Ensure images are sent to the GPU
            val_outputs = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=args.overlap, mode='gaussian', sw_device="cuda", device="cuda")
            val_outputs = F.softmax(val_outputs, dim=1)
            hard_val_outputs = torch.argmax(val_outputs, dim=1).unsqueeze(1)

        batch["pred"] = hard_val_outputs
        batch = invert_transform('pred', batch, val_transforms)
        pred = batch[0]['pred'].cpu().numpy()[0]    
        file_path_pattern = os.path.join(case_save_path, "combined_labels.nii.gz")
        align_pred = np.where(np.isin(pred, [2,3,4]), pred+1, pred) # change labels for pred: 1,2,3,4 -> 1,3,4,5
        nib.save(
                nib.Nifti1Image(align_pred.astype(np.uint8), original_affine), file_path_pattern
        )
        
        # save probabilities for each class
        if args.saveprobabilities:
            probabilities_save_path = os.path.join(save_dir, name[0].split('/')[0], 'probabilities')
            if not os.path.isdir(probabilities_save_path):
                os.makedirs(probabilities_save_path)
            for k in range(1, args.num_class):
                class_name = selected_class_map[k]
                pred_class_prob = val_outputs[0,k].cpu().numpy()
                pred_class_prob = (pred_class_prob * 255).astype(np.uint8) # convert to 0-255
                file_path_pattern = os.path.join(probabilities_save_path, f"{class_name}.nii.gz")
                nib.save(
                        nib.Nifti1Image(pred_class_prob, original_affine), file_path_pattern
                )            
        
        if not os.path.isdir(organ_seg_save_path):
            os.makedirs(organ_seg_save_path)
        for k in range(1, args.num_class):
            class_name = selected_class_map[k]
            if class_name == 'pancreas':
                pred_class = np.where(np.isin(pred, [2,3,4]), 1, pred)
                file_path_pattern = os.path.join(organ_seg_save_path, f"{class_name}.nii.gz")
                nib.save(
                        nib.Nifti1Image(pred_class.astype(np.uint8), original_affine), file_path_pattern
                )
            else:   
                pred_class = (pred == k)
                file_path_pattern = os.path.join(organ_seg_save_path, f"{class_name}.nii.gz")
                nib.save(
                        nib.Nifti1Image(pred_class.astype(np.uint8), original_affine), file_path_pattern
                )
        
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0,type = int)
    ## logging
    parser.add_argument('--save_dir', default='...', help='The dataset save path')
    ## model load
    parser.add_argument('--checkpoint', default='...', help='The path of trained checkpoint')
    parser.add_argument('--pretrain', default='...', 
                        help='The path of pretrain model')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')

    ## dataset
    parser.add_argument('--data_root_path', default='...', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--dataset_list', nargs='+', default=['JHH'])
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
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
    parser.add_argument('--num_class', default=25, type=int, help='class number')
    parser.add_argument('--map_type', default='vertebrae', help='class map type')
    parser.add_argument('--overlap', default=0.75, type=float, help='overlap')
    parser.add_argument('--copy_ct', action="store_true", default=False, help='copy ct file to the save_dir')

    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument('--original_label',action="store_true",default=False,help='whether dataset has original label')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')
    parser.add_argument('--cpu',action="store_true", default=False, help='The entire inference process is performed on the GPU ')
    parser.add_argument('--threshold_organ', default='Pancreas Tumor')
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet]')
    parser.add_argument('--create_dataset',action="store_true", default=False)
    parser.add_argument('--saveprobabilities',action="store_true", default=False, help='save the probabilities of the model')
    parser.add_argument('--stage', default='test', help='train or test')

    args = parser.parse_args()

    # prepare the 3D model
    
    if args.backbone == 'segresnet':
        model = SegResNet(
                    blocks_down=[1, 2, 2, 4],
                    blocks_up=[1, 1, 1],
                    init_filters=16,
                    in_channels=1,
                    out_channels=args.num_class,
                    dropout_prob=0.0,
                    )
        if '.0422.' in args.checkpoint:
            store_dict = model.state_dict()
            model_dict = torch.load(args.checkpoint, map_location='cpu')['net']
            new_model_dict={}
            for key, value in model_dict.items():
                new_key = key.replace('module.', '')
                new_model_dict[key] = value
            model_dict = new_model_dict  
            amount = 0
            for key in model_dict.keys():
                new_key = '.'.join(key.split('.')[1:])
                if new_key in store_dict.keys():
                    store_dict[new_key] = model_dict[key]   
                    amount += 1
            assert amount == len(store_dict), "the model is not loaded successfully"
        else:
            store_dict = model.state_dict()
            model_dict = torch.load(args.checkpoint, map_location='cpu')['net']
            amount = 0
            for key in model_dict.keys():
                store_dict[key] = model_dict[key]   
                amount += 1
            assert amount == len(store_dict), "the model is not loaded successfully"
            
    if args.backbone == 'swinunetr':
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
        model_dict = torch.load(args.checkpoint, map_location='cpu')['net']
        new_model_dict={}
        for key, value in model_dict.items():
            new_key = key.replace('module.', '')
            new_model_dict[key] = value
        model_dict = new_model_dict  
        amount = 0
        for key in model_dict.keys():
            store_dict[key] = model_dict[key]   
            amount += 1
        assert amount == len(store_dict), "the model is not loaded successfully"

    model.load_state_dict(store_dict)
    model.cuda()
    torch.backends.cudnn.benchmark = True
    test_loader, test_transforms = get_loader(args)
    validation(model, test_loader, test_transforms, args)

if __name__ == "__main__":
    main()