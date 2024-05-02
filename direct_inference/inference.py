import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import argparse
import shutil
import nibabel as nib
from monai.inferers import sliding_window_inference
from model.Universal_model import Universal_model
from model.SwinUNETR_target import SwinUNETR
from dataset.dataloader_test import get_loader, taskmap_set
from utils.utils import threshold_organ, pseudo_label_all_organ, pseudo_label_single_organ
from utils.utils import TEMPLATE, NUM_CLASS,ORGAN_NAME_LOW
from utils.utils import organ_post_process, threshold_organ, invert_transform

torch.multiprocessing.set_sharing_strategy('file_system')


def validation(model, ValLoader, val_transforms, args):
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model.eval()
    if args.suprem:
        dice_list = {}
        for key in TEMPLATE.keys():
            dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
        for index, batch in enumerate(tqdm(ValLoader)):
            
            image,name_img = batch["image"].cuda(),batch["name_img"]
            image_file_path = os.path.join(args.data_root_path,name_img[0], 'ct.nii.gz')
            case_save_path = os.path.join(save_dir,name_img[0].split('/')[0])
            print(case_save_path)
            if not os.path.isdir(case_save_path):
                os.makedirs(case_save_path)
            organ_seg_save_path = os.path.join(save_dir,name_img[0].split('/')[0],'segmentations')
            print(image_file_path)
            print(image.shape)
            print(name_img)
            if args.copy_ct:
                destination_ct = os.path.join(case_save_path,'ct.nii.gz')
                if not os.path.isfile(destination_ct):
                    shutil.copy(image_file_path, destination_ct)
                    print("CT scans copied successfully.")
            affine_temp = nib.load(image_file_path).affine
            with torch.no_grad():
                pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=args.overlap, mode='gaussian')
                pred_sigmoid = F.sigmoid(pred)
            pred_hard = threshold_organ(pred_sigmoid,args)
            pred_hard = pred_hard.cpu()
            torch.cuda.empty_cache()

            B = pred_hard.shape[0]
            for b in range(B):
                organ_list_all = TEMPLATE['target'] # post processing target organ
                pred_hard_post, _ = organ_post_process(pred_hard.numpy(), organ_list_all, case_save_path,args)
                pred_hard_post = torch.tensor(pred_hard_post)
            
            if args.store_result:
                if not os.path.isdir(organ_seg_save_path):
                    os.makedirs(organ_seg_save_path)
                organ_index_all = TEMPLATE['target']
                for organ_index in organ_index_all:
                    pseudo_label_single = pseudo_label_single_organ(pred_hard_post,organ_index,args)
                    organ_name = ORGAN_NAME_LOW[organ_index-1]
                    batch[organ_name]=pseudo_label_single.cpu()
                    BATCH = invert_transform(organ_name,batch,val_transforms)
                    organ_invertd = np.squeeze(BATCH[0][organ_name].numpy(),axis = 0)
                    # save organ labels as the np.int8 type
                    organ_inverted_type=organ_invertd.astype(np.uint8)
                    organ_save = nib.Nifti1Image(organ_inverted_type,affine_temp)
                    new_name = os.path.join(organ_seg_save_path, organ_name+'.nii.gz')
                    print('organ seg saved in path: %s'%(new_name))
                    nib.save(organ_save,new_name)
                    
                pseudo_label_all = pseudo_label_all_organ(pred_hard_post,args)
                batch['pseudo_label'] = pseudo_label_all.cpu()
                BATCH = invert_transform('pseudo_label',batch,val_transforms)
                pseudo_label_invertd = np.squeeze(BATCH[0]['pseudo_label'].numpy(),axis = 0)
                pseudo_label_inverted_type=pseudo_label_invertd.astype(np.uint8)
                pseudo_label_save = nib.Nifti1Image(pseudo_label_inverted_type,affine_temp)
                new_name = os.path.join(case_save_path, 'combined_labels.nii.gz')
                nib.save(pseudo_label_save,new_name)
                print('pseudo label saved in path: %s'%(new_name))
        
    if args.customize:
        selected_class_map = taskmap_set[args.map_type]
        count = 0
        for index, batch in enumerate(tqdm(ValLoader)):
            image, name = batch["image"].to(args.device), batch["name_img"]
            image_file_path = os.path.join(args.data_root_path,name[0], 'ct.nii.gz')
            case_save_path = os.path.join(save_dir,name[0].split('/')[0])
            print(case_save_path)
            if not os.path.isdir(case_save_path):
                os.makedirs(case_save_path)
            organ_seg_save_path = os.path.join(save_dir,name[0].split('/')[0],'segmentations')
            print(image_file_path)
            print(image.shape)
            print(name)
            if args.copy_ct:
                destination_ct = os.path.join(case_save_path,'ct.nii.gz')
                if not os.path.isfile(destination_ct):
                    shutil.copy(image_file_path, destination_ct)
                    print("CT scans copied successfully.")         
            name = name.item() if isinstance(name, torch.Tensor) else name  # Convert to Python str if it's a Tensor
            original_affine = nib.load(image_file_path).affine
            with torch.no_grad():
                # print("Image: {}, shape: {}".format(name[0], image.shape))
                val_outputs = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=args.overlap, mode='gaussian', sw_device="cuda", device="cpu")
                val_outputs = val_outputs.sigmoid()
                # print(val_outputs.shape)
                hard_val_outputs = torch.argmax(val_outputs, dim=1).unsqueeze(1)
                # print(hard_val_outputs.shape)
                # print(np.unique(hard_val_outputs))
 
            batch["pred"] = hard_val_outputs
            batch = invert_transform('pred', batch, val_transforms)
            pred = batch[0]['pred'].cpu().numpy()[0]
            # print(pred.shape)    
            file_path_pattern = os.path.join(case_save_path, "combined_labels.nii.gz")
            nib.save(
                    nib.Nifti1Image(pred.astype(np.uint8), original_affine), file_path_pattern
            )
            if not os.path.isdir(organ_seg_save_path):
                os.makedirs(organ_seg_save_path)
            for k in range(1, args.num_class):
                class_name = selected_class_map[k]
                pred_class = (pred == k)
                file_path_pattern = os.path.join(organ_seg_save_path, f"{class_name}.nii.gz")
                nib.save(
                        nib.Nifti1Image(pred_class.astype(np.uint8), original_affine), file_path_pattern
                )
            count += 1
            print("[{}/{}] Saved {}".format(count,len(ValLoader),name[0]))
            
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', action="store_true", default=False,
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
    parser.add_argument('--suprem',action="store_true", default=False)
    parser.add_argument('--customize',action="store_true", default=False)

    args = parser.parse_args()

    # prepare the 3D model
    
    if args.suprem:
        model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                        in_channels=1,
                        out_channels=NUM_CLASS,
                        backbone=args.backbone,
                        encoding='word_embedding'
                        )
        #Load pre-trained weights
        store_dict = model.state_dict()
        store_dict_keys = [key for key, value in store_dict.items()]
        checkpoint = torch.load(args.checkpoint)
        load_dict = checkpoint['net']
        load_dict_value = [value for key, value in load_dict.items()]

        for i in range(len(store_dict)):
            store_dict[store_dict_keys[i]] = load_dict_value[i]
    
    if args.customize:
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
        model_dict = torch.load(args.checkpoint)['net']
        store_dict = model.state_dict()
        amount = 0
        for key in model_dict.keys():
            new_key = '.'.join(key.split('.')[1:])
            if new_key in store_dict.keys():
                store_dict[new_key] = model_dict[key]   
                amount += 1
        print(amount, len(store_dict.keys()))
        
    model.load_state_dict(store_dict)
    print('Use pretrained weights')
    model.cuda()
    torch.backends.cudnn.benchmark = True
    test_loader, val_transforms = get_loader(args)
    validation(model, test_loader, val_transforms, args)

if __name__ == "__main__":
    main()
