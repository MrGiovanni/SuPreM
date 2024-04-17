'''
python -W ignore plot_website_video_standard_color_multiprocessing.py --abdomen_atlas /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo --png_save_path /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo1.0PNG --video_save_path /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo1.0AVI --gif_save_path /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo1.0GIF --FPS 25

python -W ignore plot_website_video_standard_color_multiprocessing.py --abdomen_atlas /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo --png_save_path /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo1.1PNG --video_save_path /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo1.1AVI --gif_save_path /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo1.1GIF --FPS 25
'''

import numpy as np 
import os
import cv2
import argparse
import copy
import nibabel as nib
from tqdm import tqdm 
from PIL import Image
import imageio
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

low_range = -150
high_range = 250

# AbdomenAtlas 1.0
class_of_interest = ['aorta',
                     'gall_bladder',
                     'kidney_left',
                     'kidney_right',
                     'liver',
                     'pancreas',
                     'postcava',
                     'spleen',
                     'stomach',
                    ]

# JHH Dataset
# class_of_interest = ['aorta',
#                      'gallbladder',
#                      'kidney_left',
#                      'kidney_right',
#                      'liver',
#                      'pancreas',
#                      'postcava',
#                      'spleen',
#                      'stomach',
#                      'adrenal_gland_right',
#                      'adrenal_gland_left',
#                      'duodenum',
#                      'intestine',
#                      'colon',
#                     ]

# AbdomenAtlas 1.1
# class_of_interest = ['aorta',
#                      'gall_bladder',
#                      'kidney_left',
#                      'kidney_right',
#                      'liver',
#                      'pancreas',
#                      'postcava',
#                      'spleen',
#                      'stomach',
#                      'adrenal_gland_left',
#                      'adrenal_gland_right',
#                      'bladder',
#                      'celiac_trunk',
#                      'colon',
#                      'duodenum',
#                      'esophagus',
#                      'femur_left',
#                      'femur_right',
#                      'hepatic_vessel',
#                      'intestine',
#                      'lung_left',
#                      'lung_right',
#                      'portal_vein_and_splenic_vein',
#                      'prostate',
#                      'rectum'
#                     ]

CLASS_IND = {
    'spleen': 1,
    'kidney_right': 2,
    'kidney_left': 3,
    'gall_bladder': 4,
    'esophagus': 5,
    'liver': 6,
    'stomach': 7,
    'aorta': 8,
    'postcava': 9,
    'portal_vein_and_splenic_vein': 10,
    'pancreas': 11,
    'adrenal_gland_right': 12,
    'adrenal_gland_left': 13,
    'duodenum': 14,
    'hepatic_vessel': 15,
    'lung_right': 16,
    'lung_left': 17,
    'colon': 18,
    'intestine': 19,
    'rectum': 20,
    'bladder': 21,
    'prostate': 22,
    'femur_left': 23,
    'femur_right': 24,
    'celiac_trunk': 25,
    'kidney_tumor': 26,
    'liver_tumor': 27,
    'pancreatic_tumor': 28,
    'hepatic_vessel_tumor': 29,
    'lung_tumor': 30,
    'colon_tumor': 31,
    'kidney cyst': 32,
}

CLASS_RGB_OPACITY = {
    'spleen': ((157, 108, 162), 1.0),
    'kidney_right': ((185, 102, 83), 1.0),
    'kidney_left': ((185, 102, 83), 1.0),
    'gall_bladder': ((139, 150, 98), 1.0),
    'esophagus': ((211, 171, 143), 1.0),
    'liver': ((221, 130, 101), 1.0),
    'stomach': ((216, 132, 105), 0.7),
    'aorta': ((242, 86, 67), 1.0),
    'postcava': ((67, 152, 206), 1.0),
    'portal_vein_and_splenic_vein': ((67, 152, 206), 1.0),
    'pancreas': ((243, 179, 110), 1.0),
    'adrenal_gland_right': ((244, 186, 150), 1.0),
    'adrenal_gland_left': ((244, 186, 150), 1.0),
    'duodenum': ((255, 253, 229), 1.0),
    'hepatic_vessel': 15,
    'lung_right': ((197, 164, 145), 1.0),
    'lung_left': ((197, 164, 145), 1.0),
    'colon': ((204, 168, 143), 1.0),
    'intestine': 19,
    'rectum': 20,
    'bladder': ((222, 154, 132), 1.0),
    'prostate': ((230, 158, 140), 1.0),
    'femur_left': 23,
    'femur_right': 24,
    'celiac_trunk': 25,
    'kidney_tumor': 26,
    'liver_tumor': 27,
    'pancreatic_tumor': 28,
    'hepatic_vessel_tumor': 29,
    'lung_tumor': 30,
    'colon_tumor': 31,
    'kidney_cyst': 32,
}


def add_colorful_mask(image, mask, class_index):

    for class_name in class_of_interest:

        indices = mask == class_index[class_name]
        color, opacity = CLASS_RGB_OPACITY[class_name][0], CLASS_RGB_OPACITY[class_name][1]
        image[indices, 0] = np.clip(image[indices, 0] * (1 - opacity) + color[0] * opacity, 0, 255)
        image[indices, 1] = np.clip(image[indices, 0] * (1 - opacity) + color[1] * opacity, 0, 255)
        image[indices, 2] = np.clip(image[indices, 0] * (1 - opacity) + color[2] * opacity, 0, 255)
    
    return image


def load_individual_maps(segmentation_dir):
    # Initializes an empty mask array to hold the combined masks.
    combined_mask = None
    
    for c in class_of_interest:
        mask_path = os.path.join(segmentation_dir, c + '.nii.gz')
        c_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        
        if combined_mask is None:
            combined_mask = np.zeros_like(c_mask)
        
        combined_mask[c_mask == 1] = CLASS_IND[c]
    
    return combined_mask


def find_roi_bounds(mask):
    # Identifies the top and bottom z-values with masks for the ROI.
    present_slices = np.any(mask, axis=(0, 1))
    z_min, z_max = np.where(present_slices)[0][[0, -1]]
    return z_min + 5, z_max - 5

def full_make_png(case_name, args):
    
    for plane in ['axial', 'coronal', 'sagittal']:
        if not os.path.exists(os.path.join(args.png_save_path, plane, case_name)):
            os.makedirs(os.path.join(args.png_save_path, plane, case_name))

    image_name = f'ct.nii.gz'

    image_path = os.path.join(args.abdomen_atlas, case_name, image_name)

    # single case
    image = nib.load(image_path).get_fdata().astype(np.int16)
    # Load mask and find ROI bounds.
    mask = load_individual_maps(os.path.join(args.abdomen_atlas, case_name, 'segmentations'))
    z_min, z_max = find_roi_bounds(mask)

    # Modifications to handle the selected ROI instead of the whole volume.
    for z in range(z_min, z_max + 1):
        # Generate and save images for the ROI only.
        pass  # Image generation and saving code here, adjusted for ROI.
    
    # change orientation
    ornt = nib.orientations.axcodes2ornt(('F', 'L', 'U'), (('L','R'),('B','F'),('D','U')))
    image = nib.orientations.apply_orientation(image, ornt)
    mask = nib.orientations.apply_orientation(mask, ornt)
    
    image[image > high_range] = high_range
    image[image < low_range] = low_range
    image = np.round((image - low_range) / (high_range - low_range) * 255.0).astype(np.uint8)
    image = np.repeat(image.reshape(image.shape[0],image.shape[1],image.shape[2],1), 3, axis=3)
    original_image = copy.deepcopy(image)
    
    image_mask = add_colorful_mask(image, mask, CLASS_IND)
    
    concat_frame = np.concatenate((original_image, image_mask), axis = 1)
    
    for z in range(mask.shape[2]):
        Image.fromarray(concat_frame[:,:,z,:]).save(os.path.join(args.png_save_path, 'axial', case_name, str(z)+'.png'))

    for z in range(mask.shape[1]):
        Image.fromarray(concat_frame[:,z,:,:]).save(os.path.join(args.png_save_path, 'sagittal', case_name, str(z)+'.png'))

    for z in range(mask.shape[0]):
        Image.fromarray(concat_frame[z,:,:,:]).save(os.path.join(args.png_save_path, 'coronal', case_name, str(z)+'.png'))
        
def make_avi(case_name, plane, args):

    if not os.path.exists(os.path.join(args.video_save_path, plane)):
        os.makedirs(os.path.join(args.video_save_path, plane))
    if not os.path.exists(os.path.join(args.gif_save_path, plane)):
        os.makedirs(os.path.join(args.gif_save_path, plane))
    
    image_folder = os.path.join(args.png_save_path, plane, case_name)
    video_name = os.path.join(args.video_save_path, plane, case_name+'.avi')
    gif_name = os.path.join(args.gif_save_path, plane, case_name+'.gif')
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    for i in range(len(images)):
        images[i] = images[i].replace('.png','')
        images[i] = int(images[i])
    images.sort()

    frame = cv2.imread(os.path.join(image_folder, str(images[0])+'.png'))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, args.FPS, (width,height))

    imgs = []
    for image in images:
        img = cv2.imread(os.path.join(image_folder, str(image)+'.png'))
        video.write(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    cv2.destroyAllWindows()
    video.release()
    imageio.mimsave(gif_name, imgs, duration=args.FPS*0.4)
    
def event(folder, args):
    if folder == '.ipynb_checkpoints':
        return
    full_make_png(folder, args)
    for plane in ['axial', 'coronal', 'sagittal']:
        make_avi(folder, plane, args)
        
def main(args):
    all_folder_names = [name for name in os.listdir(args.abdomen_atlas) if os.path.isdir(os.path.join(args.abdomen_atlas, name))]
    
    folder_names = []
    for pid in tqdm(all_folder_names):
        mask_path = os.path.join(args.abdomen_atlas, pid, 'segmentations', 'liver.nii.gz')
        mask = nib.load(mask_path)
        mask_shape = mask.header['dim']
        if mask_shape[3] > args.minimal_slices:
            folder_names.append(pid)

    print('>> {} CPU cores are secured.'.format(int(cpu_count()*0.8)))
    
    with ProcessPoolExecutor(max_workers=int(cpu_count()*0.8)) as executor:
        futures = {executor.submit(event, folder, args): folder
                   for folder in folder_names}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            folder = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {folder}: {e}")
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--abdomen_atlas', dest='abdomen_atlas', type=str, default='/Volumes/Atlas/AbdomenAtlas_8K_internal',
                        help='the directory of the AbdomenAtlas dataset',
                       )
    parser.add_argument("--png_save_path", dest='png_save_path', type=str, default='./materials',
                        help='the directory of png for each CT slice',
                       )
    parser.add_argument("--video_save_path", dest='video_save_path', type=str, default='./videos',
                        help='the directory for saving videos',
                       )
    parser.add_argument("--gif_save_path", dest='gif_save_path', type=str, default='./gifs',
                        help='the directory for saving gifs',
                       )
    parser.add_argument("--minimal_slices", dest='minimal_slices', type=int, default=500,
                        help='the minimal value for the number of CT slices',
                       )
    parser.add_argument("--FPS", dest='FPS', type=float, default=20,
                        help='the FPS value for videos; larger the value, faster the video',
                       )
    args = parser.parse_args()

    if not os.path.exists(args.png_save_path):
        os.makedirs(args.png_save_path)

    if not os.path.exists(args.video_save_path):
        os.makedirs(args.video_save_path)

    if not os.path.exists(args.gif_save_path):
        os.makedirs(args.gif_save_path)
    
    main(args)
