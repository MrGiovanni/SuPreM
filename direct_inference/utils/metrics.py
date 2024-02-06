import torch
import numpy as np
from scipy import ndimage

def dice_score(preds, labels, spe_sen=False):  # on GPU
    ### preds: w,h,d; label: w,h,d
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    preds = torch.where(preds > 0.5, 1., 0.)
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)

    tp = torch.sum(torch.mul(predict, target))
    fn = torch.sum(torch.mul(predict!=1, target))
    fp = torch.sum(torch.mul(predict, target!=1))
    tn = torch.sum(torch.mul(predict!=1, target!=1))

    den = torch.sum(predict) + torch.sum(target) + 1

    dice = 2 * tp / den
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn/(fp + tn)

    if spe_sen:
        return dice, recall, precision, specificity
    else:
        return dice, recall, precision
    
def calculate_metrics(attention_ideal,attention_real):
    ## organ_metrics_data: attention/overlap/uncertainty

    tp = np.sum(np.multiply(attention_ideal,attention_real),axis = (0,1,2))
    fp = np.sum(np.multiply(attention_ideal!=1,attention_real),axis = (0,1,2))
    fn = np.sum(np.multiply(attention_ideal,attention_real!=1),axis = (0,1,2))
    tn = np.sum(np.multiply(attention_ideal!=1,attention_real!=1),axis = (0,1,2))


    sensitivity = tp/(tp+fn)
    
    specificity= tn/(tn+fp)
    
    precision = tp/(tp+fp)
    
    return sensitivity,specificity,precision

def calculate_dice(mask1,mask2):
    intersection = np.sum(mask1*mask2)
    sum_masks = np.sum(mask1)+np.sum(mask2)
    smooth = 1e-4
    dice = (2.*intersection+smooth)/(sum_masks+smooth)
    return dice

def get_mask_edges(mask):
    b_mask = mask ==1
    # convert to bool array
    if isinstance(b_mask, torch.Tensor):
        b_mask = b_mask.cpu().numpy()  # Move to CPU and convert to numpy if it's a CUDA tensor
    edges = ndimage.binary_erosion(b_mask) ^ b_mask
    return edges

def get_surface_distance(mask1,mask2,spacing):
    edges1 = get_mask_edges(mask1)
    edges2 = get_mask_edges(mask2)
    dis = ndimage.distance_transform_edt(~edges1, sampling=spacing)
    return np.asarray(dis[edges2])

def surface_dice(mask1,mask2,spacing,tolerance):
    dis1 = get_surface_distance(mask1,mask2,spacing)
    dis2 = get_surface_distance(mask2,mask1,spacing)
    boundary_complete = len(dis1) + len(dis2)
    boundary_correct = np.sum(dis1 <= tolerance) + np.sum(dis2 <= tolerance)
    nsd = boundary_correct / boundary_complete
    return nsd

