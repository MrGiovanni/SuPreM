import torch
import numpy as np

import os
import argparse
import time
import csv
import nibabel as nib
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")
from monai.transforms import AsDiscrete
from utils.utils import dice_score, surface_dice
from dataset.dataloader import taskmap_set, weight_dice, weight_nsd
from multiprocessing import Pool, Manager

def process(args):
    ## test dict part
    test_pred = []
    test_lbl = []
    test_name = []
    for line in open(args.data_txt_path + 'test_1200case.txt'):
        name = line.strip().split('\t')[0]
        test_pred.append(args.pred_path + name + '_pred.nii.gz')
        test_lbl.append(args.dataset_path + name + '/segmentations/')
        test_name.append(name)
    data_dicts_test = [{'pred': pred, 'label': label, 'name': name}
                for pred, label, name in zip(test_pred, test_lbl, test_name)]
    print('test len {}'.format(len(data_dicts_test)))
    # data_dicts_test = data_dicts_test[:4]
    mean_wdsc, mean_wnds = validation(args, data_dicts_test)
    print("Weighted Mean DSC is:", mean_wdsc)
    print("Weighted Mean NSD is:", mean_wnds)

def computer_metrics(data_dict,organ_dice_results, organ_nsd_results, map_type):
    print('data_dict', data_dict['label'])
    post_label = AsDiscrete(to_onehot=26)

    csv_dsc = open(os.path.join(output_directory, f'{map_type}_dice_results.csv'), 'a')
    fieldnames = ["name"] + list(selected_class_map.values())
    csv_dsc_writer = csv.DictWriter(csv_dsc, fieldnames=fieldnames)

    csv_nsd = open(os.path.join(output_directory, f'{map_type}_nsd_results.csv'), 'a')
    csv_nsd_writer = csv.DictWriter(csv_nsd, fieldnames=fieldnames)

    case_name = data_dict['name']
    start_time = time.time()
    pred = nib.load(data_dict['pred'])
    pred = pred.get_fdata() 
    shape = pred.shape
    
    lbl, spacing_mm = read_label(data_dict['label'], selected_class_map, shape)
    lbl = post_label(lbl[np.newaxis,:])
    pred = post_label(pred[np.newaxis,:])
    
    dice_case_result = {"name": case_name}
    nsd_case_result = {"name": case_name}
    for class_idx, class_name in zip(selected_class_map.keys(), selected_class_map.values()):
        # dice, nsd = cal_dice_nsd(pred[class_idx], lbl[class_idx], spacing_mm, 1)  # unpack the returned tuple and only take the dice score
        dice, _, _ = dice_score(torch.from_numpy(pred[class_idx]), torch.from_numpy(lbl[class_idx]))  # unpack the returned tuple and only take the dice score
        dice = dice.item() if torch.is_tensor(dice) else dice  # convert tensor to Python native data type if it's a tensor
        nsd = surface_dice(torch.from_numpy(pred[class_idx]), torch.from_numpy(lbl[class_idx]), spacing_mm, 1)  # using retrieved spacing here
        organ_lbl = ndimage.binary_erosion(lbl[class_idx], structure=np.ones((3,3,3)), iterations=1)
        if np.sum(organ_lbl)>10:

            dice_case_result[class_name] = round(dice, 3)
            tmp=organ_dice_results[class_name]
            tmp.append(round(dice, 3))
            organ_dice_results.update({class_name:tmp})

            nsd_case_result[class_name] = round(nsd, 3)
            tmp=organ_nsd_results[class_name]
            tmp.append(round(nsd, 3))
            organ_nsd_results.update({class_name:tmp})
        else:
            dice_case_result[class_name] = np.NaN 
            nsd_case_result[class_name] = np.NaN
    print('total_time: ', time.time()-start_time) 
    print('dice_case_result', dice_case_result)

    csv_dsc_writer.writerows([dice_case_result])
    csv_nsd_writer.writerows([nsd_case_result])
    csv_dsc.close()
    csv_nsd.close()

    
def validation(args, data_dicts_test):
    selected_class_map = taskmap_set[args.map_type]

    organ_dice_results = Manager().dict()
    organ_nsd_results = Manager().dict()
    for i in selected_class_map.values():
        organ_dice_results[i] = []
        organ_nsd_results[i] = []

    csv_dsc = open(os.path.join(output_directory, f'{args.map_type}_dice_results.csv'), 'a')
    fieldnames = ["name"] + list(selected_class_map.values())
    csv_dsc_writer = csv.DictWriter(csv_dsc, fieldnames=fieldnames)
    csv_dsc_writer.writeheader()
    csv_dsc.close()

    csv_nsd = open(os.path.join(output_directory, f'{args.map_type}_nsd_results.csv'), 'a')
    csv_nsd_writer = csv.DictWriter(csv_nsd, fieldnames=fieldnames)
    csv_nsd_writer.writeheader()
    csv_nsd.close()

    pool = Pool(processes=args.num_workers)
    for index, batch in enumerate(data_dicts_test):
        pool.apply_async(computer_metrics, (batch,organ_dice_results, organ_nsd_results, args.map_type))
    pool.close()
    pool.join()
    print('organ_dice_results', organ_dice_results)

    avg_dsc = {"name": "avg"}
    avg_nsd = {"name": "avg"}
    wavg_dsc = {"name": "weighted avg"}
    wavg_nsd = {"name": "weighted avg"}
    for i in organ_dice_results.keys():
        avg_dsc.update({i:round(np.array(organ_dice_results[i]).mean(), 3) }) 
        avg_nsd.update({i:round(np.array(organ_nsd_results[i]).mean(), 3)})
        wavg_dsc.update({i:round(np.array(organ_dice_results[i]).mean()*weight_dice[i], 3)})
        wavg_nsd.update({i:round(np.array(organ_nsd_results[i]).mean()*weight_nsd[i], 3)})
        
    csv_dsc = open(os.path.join(output_directory, f'{args.map_type}_dice_results.csv'), 'a')
    fieldnames = ["name"] + list(selected_class_map.values())
    csv_dsc_writer = csv.DictWriter(csv_dsc, fieldnames=fieldnames)

    csv_nsd = open(os.path.join(output_directory, f'{args.map_type}_nsd_results.csv'), 'a')
    csv_nsd_writer = csv.DictWriter(csv_nsd, fieldnames=fieldnames)
    
    csv_dsc_writer.writerows([avg_dsc])
    csv_nsd_writer.writerows([avg_nsd])
    csv_dsc_writer.writerows([wavg_dsc])
    csv_nsd_writer.writerows([wavg_nsd])
    csv_dsc.close()
    csv_nsd.close()
    
    
    # calculate weighted mDSC & weighted mNSD
    wavg_dsc_value = [wavg_dsc[i] for i in wavg_dsc.keys() if i != "name" and not np.isnan(wavg_dsc[i])]
    wavg_nsd_value = [wavg_nsd[i] for i in wavg_nsd.keys() if i != "name" and not np.isnan(wavg_nsd[i])]
    wmean_dsc = round(np.array(wavg_dsc_value).sum(), 3) 
    wmean_nsd = round(np.array(wavg_nsd_value).sum(), 3)  
    
    # calculate mDSC & weighted mNSD
    avg_dsc_value = [avg_dsc[i] for i in avg_dsc.keys() if i != "name" and not np.isnan(avg_dsc[i])]
    avg_nsd_value = [avg_nsd[i] for i in avg_nsd.keys() if i != "name" and not np.isnan(avg_nsd[i])]
    mean_dsc = round(np.array(avg_dsc_value).mean(), 3) 
    mean_nsd = round(np.array(avg_nsd_value).mean(), 3)  

    with open(os.path.join(output_directory, 'scores.txt'),'a+') as score_file:
        score_file.writelines("wmDSC wmNSD")
        score_file.writelines("\n")
        score_file.writelines(" ".join([str(wmean_dsc), str(wmean_nsd)]))
        score_file.writelines("\n")
        score_file.writelines("mDSC mNSD")
        score_file.writelines("\n")
        score_file.writelines(" ".join([str(mean_dsc), str(mean_nsd)]))
        
    return wmean_dsc, wmean_nsd

def read_label(lbl_dir, organ_map, shape):
    organ_lbl = np.zeros(shape)

    for index, organ in organ_map.items():
        if 'lung_left' == organ:
            array = nib.load(lbl_dir + 'lung_upper_lobe_left' + '.nii.gz').get_fdata() 
            organ_lbl[array == 1] = index
            array = nib.load(lbl_dir + 'lung_lower_lobe_left' + '.nii.gz').get_fdata() 
            organ_lbl[array == 1] = index

        elif 'lung_right' == organ:
            array = nib.load(lbl_dir + 'lung_upper_lobe_right' + '.nii.gz').get_fdata() 
            organ_lbl[array == 1] = index
            array = nib.load(lbl_dir + 'lung_middle_lobe_right' + '.nii.gz').get_fdata() 
            organ_lbl[array == 1] = index
            array = nib.load(lbl_dir + 'lung_lower_lobe_right' + '.nii.gz').get_fdata() 
            organ_lbl[array == 1] = index
            
        else:
            if index == 1:
                array = nib.load(lbl_dir + organ + '.nii.gz')
                pixdim = array.header['pixdim']
                spacing_mm = tuple(pixdim[1:4])
                array = array.get_fdata() 
            else:
                array = nib.load(lbl_dir + organ + '.nii.gz').get_fdata() 
            organ_lbl[array == 1] = index
        
            
    return organ_lbl, spacing_mm

parser = argparse.ArgumentParser()
## logging
parser.add_argument('--log_name', default='multiprocess', help='The path resume from checkpoint')
## dataset
parser.add_argument('--pred_path', default='.', help='dataset path')
parser.add_argument('--dataset_path', default='/mnt/ccvl15/zzhou82/PublicAbdominalData/16_TotalSegmentor/', help='dataset path')
parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
# change here
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--map_type', default='totalseg', help='sample number in each ct')
parser.add_argument('--num_class', default=25, type=int, help='class num')
args = parser.parse_args()

output_directory = os.path.join('out', args.log_name)
os.makedirs(output_directory, exist_ok=True)
selected_class_map = taskmap_set[args.map_type]




process(args=args)


# if __name__ == "__main__":
    # # python -W ignore test_totalseg.py --dataset_path xxx --pred_path xxx
    # main()


