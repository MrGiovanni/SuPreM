import os
import nibabel as nib
import numpy as np

def load_mask(pid, class_name, datapath):

    mask_path = os.path.join(datapath, pid, 'segmentations', class_name + '.nii.gz')
    if os.path.isfile(mask_path):
        return nib.load(mask_path).get_fdata().astype(np.uint8)
    else:
        return None
    
def aorta_error(pid, datapath):

    liver = load_mask(pid, 'liver', datapath)
    aorta = load_mask(pid, 'aorta', datapath)
    lung_left = load_mask(pid, 'lung_left', datapath)
    lung_right = load_mask(pid, 'lung_right', datapath)
    postcava = load_mask(pid, 'postcava', datapath)

    error_count = 0

    if (liver.shape[0] == 512 and liver.shape[1] == 512) or \
    (liver.shape[0] != liver.shape[1] and liver.shape[1] != liver.shape[2]):
        for i in range(liver.shape[-1]):
            if error2d_isliver_noaorta(liver[:,:,i], aorta[:,:,i]) or \
            error2d_ispostcava_islung_nonaorta(postcava[:,:,i], lung_right[:,:,i], lung_left[:,:,i], aorta[:,:,i]):
                error_count += 1
        error_percent = 100.0*error_count/liver.shape[-1]
        print('> {} has {:.1f}% ({}/{}) errors in aorta'.format(pid,
                                                    error_percent,
                                                    error_count,
                                                    liver.shape[-1],
                                                    ))
    
    else:
        for i in range(liver.shape[0]):
            if error2d_isliver_noaorta(liver[i,:,:], aorta[i,:,:]) or \
                error2d_ispostcava_islung_nonaorta(postcava[i,:,:], lung_right[i,:,:], lung_left[i,:,:], aorta[i,:,:]):
                error_count += 1
        error_percent = 100.0*error_count/liver.shape[0]
        print('> {} has {:.1f}% ({}/{}) errors in aorta'.format(pid,
                                                    error_percent,
                                                    error_count,
                                                    liver.shape[0],
                                                    ))
    if error_percent > 3:
        error_detected = True
    else:
        error_detected = False

    return error_detected

def kidney_error(pid, datapath):

    error_detected = False

    kidney_left = load_mask(pid, 'kidney_left', datapath)
    kidney_right = load_mask(pid, 'kidney_right', datapath)

    kidney_lr_overlap = error3d_overlaps(kidney_left, kidney_right)
    if kidney_lr_overlap > 0:
        print('> {} has {} px overlap betwee kidney L&R'.format(pid, kidney_lr_overlap))
        error_detected = True

    return error_detected

def kidney_postprocessing(pid, datapath):

    kidney_left = load_mask(pid, 'kidney_left', datapath)
    kidney_right = load_mask(pid, 'kidney_right', datapath)

def error2d_isliver_noaorta(liver, aorta):

    if liver is None or aorta is None:
        return False

    if np.sum(liver) > 0 and np.sum(aorta) == 0:
        return True
    else:
        return False
    
def error2d_ispostcava_islung_nonaorta(postcava, lung_right, lung_left, aorta):

    if postcava is None or lung_right is None or lung_left is None or aorta is None:
        return False
    
    if np.sum(postcava) > 0 and (np.sum(lung_right) > 0 or np.sum(lung_left) > 0) and np.sum(aorta) == 0:
        return True
    else:
        return False
    
def error3d_overlaps(c1, c2):

    if c1 is None or c2 is None:
        return False
    
    return np.sum(np.logical_and(c1, c2))

def error_detection_per_case(pid, args):

    if args.aorta:
        return aorta_error(pid, args.datapath)
    if args.kidney:
        return kidney_error(pid, args.datapath)