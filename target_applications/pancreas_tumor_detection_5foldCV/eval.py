'''
python eval.py --predpath /data/zzhou82/code/SuPreM/target_applications/jhh_combined_label/inference_results --truthpath /data/zzhou82/data/JHH_ROI_0.5mm
background  0
pancreas    1
pdac        3
cyst        4
pnet        5

Rules for post-processing
1. Only one (the largest connected component) pancreas. -> find_largest_connected_component()
2. The tumors must be (1) inside or next to the pancreas and (2) larger than a predefined size limit. -> find_positional_valid_tumors

Reference
https://pypi.org/project/connected-components-3d/
'''

import os
import argparse
import pandas as pd
import cc3d
import nibabel as nib
import numpy as np
from multiprocessing import Pool, Manager
from tqdm import tqdm
from scipy.ndimage import binary_dilation
import shutil
import matplotlib.pyplot as plt
from sklearn import metrics
import multiprocessing
from functools import partial

index_name_map = {
    'pancreas': 1,
    'pdac': 3,
    'cyst': 4,
    'pnet': 5,
}

def find_largest_connected_component(mask):
    labels, N = cc3d.connected_components(mask, connectivity=6, return_N=True)
    if N == 0:
        return np.zeros(mask.shape, dtype=np.uint8)  # Return an empty mask if no components
    max_label = 1 + np.argmax([np.sum(labels == i) for i in range(1, N + 1)])
    return labels == max_label

def find_positional_valid_tumors(tumor_mask, organ_mask, size_limit=10):

    pp_tumor_mask = np.zeros((tumor_mask.shape), dtype=np.uint8)

    labels_out, N = cc3d.connected_components(tumor_mask, connectivity=6, return_N=True)

    if N > 0:

        kernel = np.ones((3, 3, 3), dtype=bool)
        organ_mask_dilation = binary_dilation(organ_mask, structure=kernel)

        for segid in range(1, N+1):
            each_component = (labels_out == segid)
            if np.sum(each_component) >= size_limit:
                if np.any(each_component & organ_mask_dilation):
                    pp_tumor_mask[each_component == 1] = 1
        
    return pp_tumor_mask

def post_processing(pr, size_limits):
    pp_pr = np.zeros(pr.shape, dtype=np.uint8)
    pancreas = (pr == index_name_map['pancreas'])
    pancreas = find_largest_connected_component(pancreas)
    pp_pr[pancreas == 1] = index_name_map['pancreas']
    tumors = {'pdac': index_name_map['pdac'], 'cyst': index_name_map['cyst'], 'pnet': index_name_map['pnet']}
    for tumor, label in tumors.items():
        valid_tumors = find_positional_valid_tumors(pr == label, pancreas, size_limits[tumor])
        pp_pr[valid_tumors == 1] = label
    return pp_pr

def process_patient(args, patientID, counter):
    patient_venous = patientID + '_VENOUS'
    patient_arterial = patientID + '_ARTERIAL'
    pdac_flag = False
    cyst_flag = False
    pnet_flag = False
    gt_positive_flag = False
    if os.path.exists(os.path.join(args.predpath, patient_venous)):
        pr = nib.load(os.path.join(args.predpath, patient_venous, 'combined_labels.nii.gz')).get_fdata()
        gt_venous = nib.load(os.path.join(args.truthpath, patient_venous, 'combined_labels.nii.gz')).get_fdata()
        if args.postprocessing:
            pr = post_processing(pr, args.size_limits)
        if index_name_map['pdac'] in pr:
            pdac_flag = True
        if index_name_map['cyst'] in pr:
            cyst_flag = True
        if index_name_map['pdac'] in gt_venous or index_name_map['cyst'] in gt_venous:
            gt_positive_flag = True
    if os.path.exists(os.path.join(args.predpath, patient_arterial)):
        pr = nib.load(os.path.join(args.predpath, patient_arterial, 'combined_labels.nii.gz')).get_fdata()
        gt_arterial = nib.load(os.path.join(args.truthpath, patient_arterial, 'combined_labels.nii.gz')).get_fdata()
        if args.postprocessing:
            pr = post_processing(pr, args.size_limits)
        if index_name_map['pnet'] in pr:
            pnet_flag = True
        if index_name_map['pnet'] in gt_arterial:
            gt_positive_flag = True
                
    pr_positive_flag = False
    if pdac_flag or cyst_flag or pnet_flag:
        pr_positive_flag = True
    
    result = None
    if gt_positive_flag and pr_positive_flag:
        result = 'TP'
    elif gt_positive_flag and not pr_positive_flag:
        result = 'FN'
    elif not gt_positive_flag and pr_positive_flag:
        result = 'FP'
    elif not gt_positive_flag and not pr_positive_flag:
        result = 'TN'
    counter.increment()
    return patientID, result

class Counter(object):
    def __init__(self):
        self.val = Manager().Value('i', 0)
        self.lock = Manager().Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value

def load_pred_gt(gt_path, pred_path):
    gt_volume = nib.load(os.path.join(gt_path, 'combined_labels.nii.gz')).get_fdata()
    gt_volume = np.uint8(np.round(gt_volume))
    pred_volume = nib.load(os.path.join(pred_path, 'combined_labels.nii.gz')).get_fdata()
    pred_volume = np.uint8(np.round(pred_volume))
    W, H, D = gt_volume.shape
    pred_prob_volume = np.zeros((len(index_name_map), W, H, D))
    
    for index, (key, _) in enumerate(index_name_map.items()):
        pred_part_volume = nib.load(os.path.join(pred_path, 'probabilities', f'{key}.nii.gz')).get_fdata()
        pred_prob_volume[index] = pred_part_volume / 255.0
    return gt_volume, pred_volume, pred_prob_volume

def patient_level_tumor_detection(args, patientID):
    gt_pos_flag_temp_venous = False
    gt_pos_flag_temp_arterial = False
    pr_temp_venous = 0
    pr_temp_arterial = 0

    for phase in ['VENOUS', 'ARTERIAL']:
        specific_path = f'{patientID}_{phase}'
        if os.path.exists(os.path.join(args.predpath, specific_path)):
            gt_path = os.path.join(args.truthpath, specific_path)
            pred_path = os.path.join(args.predpath, specific_path)

            gt_volume, pr_volume, logit_volume = load_pred_gt(gt_path, pred_path)
            
            if args.postprocessing:
                pr_volume = post_processing(pr_volume, args.size_limits)

            if phase == 'VENOUS':
                pp_logit_volume = np.zeros((np.shape(pr_volume)))
                
                pancreas_volume = pr_volume == index_name_map['pancreas']
                pp_logit_volume[pancreas_volume] = (logit_volume[1][pancreas_volume] + logit_volume[2][pancreas_volume]) / 2.0
                pdac_volume = pr_volume == index_name_map['pdac']
                pp_logit_volume[pdac_volume] = logit_volume[1][pdac_volume]
                cyst_volume = pr_volume == index_name_map['cyst']
                pp_logit_volume[cyst_volume] = logit_volume[2][cyst_volume]
                
                pr_temp_venous = np.max(pp_logit_volume)

                if index_name_map['pdac'] in gt_volume or index_name_map['cyst'] in gt_volume:
                    gt_pos_flag_temp_venous = True

            if phase == 'ARTERIAL':
                pp_logit_volume = np.zeros((np.shape(pr_volume)))
                pancreas_volume = pr_volume == index_name_map['pancreas']
                pp_logit_volume[pancreas_volume] = logit_volume[3][pancreas_volume]
                pnet_volume = pr_volume == index_name_map['pnet']
                pp_logit_volume[pnet_volume] = logit_volume[3][pnet_volume]
                pr_temp_arterial = np.max(pp_logit_volume)

                if index_name_map['pnet'] in gt_volume:
                    gt_pos_flag_temp_arterial = True

    gt_pos_flag = gt_pos_flag_temp_venous or gt_pos_flag_temp_arterial
    pr = max(pr_temp_venous, pr_temp_arterial)
    
    if not os.path.exists(f'{args.saverocpath}/patientID_predprobablities_gtposneg.csv'):
        os.makedirs(args.saverocpath, exist_ok=True)
    df = pd.DataFrame({'patientID': [patientID], 'pr': [pr], 'gt_pos_flag': [gt_pos_flag]})
    df.to_csv(f'{args.saverocpath}/patientID_pr_gt_pos_flag.csv', mode='a', header=False, index=False)
    
    return (1 if gt_pos_flag else 0, pr)

def plot_roc_curve(TPR, FPR, thresholds, args,
                   linewidth=5, elinewidth=15, fontsize=80, figsize=(40,40), 
                   markersize=600, alpha=0.25,
                   zoomin=False,
                  ):
    
    baseline_sensitivity, baseline_specificity = 0.924, 0.905

    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['axes.linewidth'] = linewidth
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    plt.plot(FPR, TPR, color=[237/255,16/255,105/255], label='SuPreM', linewidth=elinewidth)
    plt.scatter(1-baseline_specificity, baseline_sensitivity, color='royalblue', label='Xia et al.', s=markersize)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    plt.axis('square')
    if zoomin:
        plt.xlim(0, 0.2); plt.xticks([0.0, 0.05, 0.1, 0.15, 0.2])
        plt.ylim(0.8, 1.0); plt.yticks([0.85, 0.9, 0.95, 1.0])
    else:
        plt.xlim(0, 1.0); plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.ylim(0.,1.0); plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    plt.grid(alpha=alpha, linewidth=linewidth)

    lg = plt.legend(['SuPreM', 
                     'Xia et al.',
                    ],
                    loc='lower right',
                    fancybox=False, shadow=False, framealpha=1.0, edgecolor='lightgray')
    lg.get_frame().set_linewidth(linewidth)

    plt.plot([1-baseline_specificity, 1-baseline_specificity], [0, 1], '--', color='royalblue', linewidth=linewidth)
    plt.plot([0, 1], [baseline_sensitivity, baseline_sensitivity], '--', color='royalblue', linewidth=linewidth)
    
    if zoomin:
        plt.text(0.15, baseline_sensitivity - 0.02, 'Sensitivity = 92.4%\nSpecificity = 90.5%', 
                fontsize=fontsize, color='royalblue',
                horizontalalignment='center', verticalalignment='center',
                )
        fig.savefig(os.path.join(args.saverocpath, 'roc_curve_zoomin.png'), 
                    bbox_inches='tight', pad_inches=0.5, dpi=200,
                    )
        fig.savefig(os.path.join(args.saverocpath, 'roc_curve_zoomin.pdf'), 
                    bbox_inches='tight', pad_inches=0.5, dpi=200,
                    )
    else:
        plt.text(0.5, baseline_sensitivity - 0.06, 'Sensitivity = 92.4%\nSpecificity = 90.5%', 
                fontsize=fontsize, color='royalblue',
                horizontalalignment='center', verticalalignment='center',
                )

        fig.savefig(os.path.join(args.saverocpath, 'roc_curve.png'), 
                    bbox_inches='tight', pad_inches=0.5, dpi=200,
                    )
        fig.savefig(os.path.join(args.saverocpath, 'roc_curve.pdf'), 
                    bbox_inches='tight', pad_inches=0.5, dpi=200,
                    )
    
    csv_file_path = os.path.join(args.saverocpath, 'roc_curve.csv') 
    data = {'TPR': TPR, 'FPR': FPR, 'thresholds': thresholds}
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)
    
def visualize_error_case(patientID, args, save_subfolder):
    if save_subfolder == 'FP':
        for phase in ['VENOUS', 'ARTERIAL']:
            patient_save_path = os.path.join(args.savevisualpath, save_subfolder, patientID, phase)
            if not os.path.exists(patient_save_path):
                os.makedirs(patient_save_path)
            if os.path.exists(os.path.join(args.predpath, f'{patientID}_{phase}')):
                pred_img = nib.load(os.path.join(args.predpath, f'{patientID}_{phase}', 'combined_labels.nii.gz'))
                gt_img = nib.load(os.path.join(args.truthpath, f'{patientID}_{phase}', 'combined_labels.nii.gz'))
                pr = post_processing(pred_img.get_fdata(), args.size_limits)
                gt = gt_img.get_fdata()
                
                pred_nii = nib.Nifti1Image(pr, pred_img.affine)
                gt_nii = nib.Nifti1Image(gt, gt_img.affine)
                
                pred_nii.set_data_dtype(np.uint8)  # save type as uint8
                gt_nii.set_data_dtype(np.uint8)  # save type as uint8

                nib.save(pred_nii, os.path.join(patient_save_path, 'prediction.nii.gz'))
                nib.save(gt_nii, os.path.join(patient_save_path, 'gt.nii.gz'))
                if args.savect:
                    shutil.copy(os.path.join(args.truthpath, f'{patientID}_{phase}', 'ct.nii.gz'), os.path.join(patient_save_path, 'ct.nii.gz'))
                    
    elif save_subfolder == 'FN':
        for phase in ['VENOUS', 'ARTERIAL']:
            patient_save_path = os.path.join(args.savevisualpath, save_subfolder, patientID, phase)
            if not os.path.exists(patient_save_path):
                os.makedirs(patient_save_path)
            if os.path.exists(os.path.join(args.predpath, f'{patientID}_{phase}')):
                gt_img = nib.load(os.path.join(args.truthpath, f'{patientID}_{phase}', 'combined_labels.nii.gz'))
                gt = gt_img.get_fdata()
                gt_nii = nib.Nifti1Image(gt, gt_img.affine)
                gt_nii.set_data_dtype(np.uint8)  # save type as uint8
                nib.save(gt_nii, os.path.join(patient_save_path, 'gt.nii.gz'))
                if args.savect:
                    shutil.copy(os.path.join(args.truthpath, f'{patientID}_{phase}', 'ct.nii.gz'), os.path.join(patient_save_path, 'ct.nii.gz'))
                    
def process_wrapper(args_tuple):
    patientID, args, save_subfolder = args_tuple
    return visualize_error_case(patientID, args, save_subfolder)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predpath', required=True, help='Path to model predictions')
    parser.add_argument('--truthpath', required=True, help='Path to ground truth')
    parser.add_argument('--postprocessing', action='store_true', help='Apply postprocessing')
    parser.add_argument('--multiprocessing', action='store_true', help='Use multiprocessing')
    parser.add_argument('--pdac_size', type=int, default=30, help='Size limit for PDAC')
    parser.add_argument('--cyst_size', type=int, default=40, help='Size limit for cyst')
    parser.add_argument('--pnet_size', type=int, default=35, help='Size limit for PNET')
    parser.add_argument('--savecsvpath', type=str, required=True, help='path to save TP, TN, FP, FN')
    parser.add_argument('--savevisualpath', type=str, required=True, help='path to save visualized error cases')
    parser.add_argument('--saverocpath', type=str, help='Path to save ROC curve', default='roc_curve')
    parser.add_argument('--FP', action='store_true', help='Process False Positives')
    parser.add_argument('--FN', action='store_true', help='Process False Negatives')
    parser.add_argument('--savect', action='store_true', help='Save CT images')
    parser.add_argument('--plotroc', action='store_true', help='Plot ROC curve')
   
    args = parser.parse_args()

    args.size_limits = {'pdac': args.pdac_size, 'cyst': args.cyst_size, 'pnet': args.pnet_size}
    
    patientIDs = set(os.path.basename(f).split('_')[0] for f in os.listdir(args.predpath) if os.path.isdir(os.path.join(args.predpath, f)))
    # exclude cases have gt errors re-reviewed by radiologists
    normal_cases = ['FELIX-Cys-1432', 'FELIX5145', 'FELIX-CYS-1289', 'FELIX7528', 'FELIX-Cys-1680', 'FELIX-Cys-1222', 'FELIX7594', 'FELIX5222', 'FELIX5224', 'FELIX-PDAC-1174', 'FELIX-Cys-1680', 'FELIX-PDAC-1174', 'FELIX7179', 'FELIX5544', 'FELIX5222', 'FELIX7521', 'FELIX-Cys-1222', 'FELIX-Cys-1632', 'FELIX-Cys-1623', 'FELIX5145', 'FELIX7521', 'FELIX-Cys-1233', 'FELIX5046', 'FELIX-Cys-1432']
    patientIDs = patientIDs - set(normal_cases)
    patientIDs = list(patientIDs)

    if args.plotroc:
        if not os.path.exists(args.saverocpath):
            os.makedirs(args.saverocpath)
        pool = multiprocessing.Pool()
        process_func = partial(patient_level_tumor_detection, args)
        results = list(tqdm(pool.imap(process_func, patientIDs), total=len(patientIDs)))

        GT, PR = zip(*results)
        fpr, tpr, thresholds = metrics.roc_curve(GT, PR)
        plot_roc_curve(tpr, fpr, thresholds, args, zoomin=False)
        plot_roc_curve(tpr, fpr, thresholds, args, zoomin=True)
        
        pool.close()
        pool.join()
    
    counter = Counter()
    if args.multiprocessing:
        with Pool() as pool:
            with tqdm(total=len(patientIDs)) as pbar:
                async_results = [pool.apply_async(process_patient, (args, id, counter)) for id in patientIDs]
                while True:
                    completed = counter.value()
                    pbar.update(completed - pbar.n)
                    if completed == len(patientIDs):
                        break
                results = [res.get() for res in async_results]
    else:
        results = []
        with tqdm(total=len(patientIDs)) as pbar:
            for id in patientIDs:
                results.append(process_patient(args, id, counter))
                pbar.update(1)

    # Save results to CSV and calculate eval_metrics
    classifications = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
    for patientID, result in results:
        if result:
            classifications[result].append(patientID)

    # Save to CSV files and calculate sensitivity, specificity, PPV
    eval_metrics = {}
    for key, ids in classifications.items():
        if not os.path.exists(args.savecsvpath):
            os.makedirs(args.savecsvpath)
        pd.DataFrame(ids, columns=['patientID']).to_csv(os.path.join(args.savecsvpath, f'{key}.csv'), index=False)
        eval_metrics[key] = len(ids)

    TP, TN, FP, FN = eval_metrics['TP'], eval_metrics['TN'], eval_metrics['FP'], eval_metrics['FN']
    sensitivity = 100 * (TP + np.finfo(float).eps) / (TP + FN + np.finfo(float).eps)
    specificity = 100 * (TN + np.finfo(float).eps) / (TN + FP + np.finfo(float).eps)
    PPV = 100 * (TP + np.finfo(float).eps) / (TP + FP + np.finfo(float).eps)

    print(f'sensitivity = {sensitivity:.2f}%')
    print(f'specificity = {specificity:.2f}%')
    print(f'PPV = {PPV:.2f}%')
    
    types_to_process = []
    if args.FP:
        types_to_process.append('FP')
    if args.FN:
        types_to_process.append('FN')

    for type_label in types_to_process:
        csv_file_path = os.path.join(args.savecsvpath, f'{type_label}.csv')
        patientIDs = pd.read_csv(csv_file_path)['patientID'].tolist()
        args_list = [(patientID, args, type_label) for patientID in patientIDs]

        if args.multiprocessing:
            with Pool() as pool:
                list(tqdm(pool.imap(process_wrapper, args_list), total=len(patientIDs)))
        else:
            for patientID, args, save_subfolder in tqdm(args_list):
                visualize_error_case(patientID, args, save_subfolder)

if __name__ == "__main__":
    main()
