import os
import nibabel as nib
import numpy as np
from skimage.measure import label

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def load_mask(pid, class_name, datapath):

    mask_path = os.path.join(datapath, pid, 'segmentations', class_name + '.nii.gz')
    if os.path.isfile(mask_path):
        nii = nib.load(mask_path)
        return nii.get_fdata().astype(np.uint8), nii.affine, nii.header
    else:
        return None, None, None

def save_mask(data, affine, header, pid, class_name, datapath):
    
    nifti_path = os.path.join(datapath, pid, 'segmentations', class_name + '.nii.gz')
    nib.save(nib.Nifti1Image(data, affine=affine, header=header), nifti_path)

def check_dim(list_of_array):

    dim = list_of_array[0].shape
    for i in range(len(list_of_array)):
        if dim != list_of_array[i].shape:
            return False
    return True

def aorta_error(pid, datapath):

    liver, _, _ = load_mask(pid, 'liver', datapath)
    aorta, _, _ = load_mask(pid, 'aorta', datapath)
    lung_left, _, _ = load_mask(pid, 'lung_left', datapath)
    lung_right, _, _ = load_mask(pid, 'lung_right', datapath)
    postcava, _, _ = load_mask(pid, 'postcava', datapath)

    assert check_dim([liver, aorta, lung_left, lung_right, postcava])

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

    kidney_left, _, _ = load_mask(pid, 'kidney_left', datapath)
    kidney_right, _, _ = load_mask(pid, 'kidney_right', datapath)

    assert check_dim([kidney_left, kidney_right])

    kidney_lr_overlap = error3d_overlaps(kidney_left, kidney_right)
    if kidney_lr_overlap > 0:
        print('> {} has {} px overlap betwee kidney L&R'.format(pid, kidney_lr_overlap))
        error_detected = True

    return error_detected

def kidney_postprocessing(pid, datapath):

    kidney_left, kidney_left_affine, kidney_left_header = load_mask(pid, 'kidney_left', datapath)
    kidney_right, kidney_right_affine, kidney_right_header = load_mask(pid, 'kidney_right', datapath)

    kidney_left = getLargestCC(kidney_left)
    kidney_right = getLargestCC(kidney_right)

    save_mask(kidney_left, kidney_left_affine, kidney_left_header, 
              pid, 'kidney_left', datapath,
              )
    save_mask(kidney_right, kidney_right_affine, kidney_right_header, 
              pid, 'kidney_right', datapath,
              )

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

