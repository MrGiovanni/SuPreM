'''
python -W ignore plot_website_video_multiprocessing.py --abdomen_atlas /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo --png_save_path /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo1.0PNG --video_save_path /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo1.0AVI --gif_save_path /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo1.0GIF --FPS 25

python -W ignore plot_website_video_multiprocessing.py --abdomen_atlas /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo --png_save_path /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo1.1PNG --video_save_path /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo1.1AVI --gif_save_path /Users/zongwei.zhou/Downloads/AbdomenAtlasDemo1.1GIF --FPS 25
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
#class_of_interest = ['aorta',
#                     'gall_bladder',
#                     'kidney_left',
#                     'kidney_right',
#                     'liver',
#                     'pancreas',
#                     'postcava',
#                     'spleen',
#                     'stomach',
#                    ]

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
class_of_interest = ['aorta',
                     'gall_bladder',
                     'kidney_left',
                     'kidney_right',
                     'liver',
                     'pancreas',
                     'postcava',
                     'spleen',
                     'stomach',
                     'adrenal_gland_left',
                     'adrenal_gland_right',
                     'bladder',
                     'celiac_trunk',
                     'colon',
                     'duodenum',
                     'esophagus',
                     'femur_left',
                     'femur_right',
                     'hepatic_vessel',
                     'intestine',
                     'lung_left',
                     'lung_right',
                     'portal_vein_and_splenic_vein',
                     'prostate',
                     'rectum'
                    ]

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

def add_colorful_mask(image, mask, class_index):
    
    image[mask == class_index['aorta'], 0] = 255   # aorta (255,0,0)
    image[mask == class_index['aorta'], 1] = image[mask == class_index['aorta'], 1] * 0.5
    image[mask == class_index['aorta'], 2] = image[mask == class_index['aorta'], 2] * 0.5
    
    image[mask == class_index['kidney_right'], 1] = 255   # kidney_right (0,255,0)
    image[mask == class_index['kidney_left'], 1] = 255   # kidney_left (0,255,0)
    image[mask == class_index['kidney_tumor'], 1] = 255  # kidney_tumor (0,255,0)
    image[mask == class_index['kidney cyst'], 1] = 255  # kidney cyst (0,255,0)

    image[mask == class_index['femur_left'], 0] = image[mask == class_index['femur_left'], 0] * 0.9  # femur (230,230,230)
    image[mask == class_index['femur_left'], 1] = image[mask == class_index['femur_left'], 1] * 0.9  # femur (230,230,230)
    image[mask == class_index['femur_left'], 2] = image[mask == class_index['femur_left'], 2] * 0.9  # femur (230,230,230)
    image[mask == class_index['femur_right'], 0] = image[mask == class_index['femur_right'], 0] * 0.9  # femur (230,230,230)
    image[mask == class_index['femur_right'], 1] = image[mask == class_index['femur_right'], 1] * 0.9  # femur (230,230,230)
    image[mask == class_index['femur_right'], 2] = image[mask == class_index['femur_right'], 2] * 0.9  # femur (230,230,230)
    
    image[mask == class_index['gall_bladder'], 0] = 255   # gallbladder (255,255,0)
    image[mask == class_index['gall_bladder'], 1] = 255   #

    image[mask == class_index['bladder'], 0] = 128   # bladder (128,255,0)
    image[mask == class_index['bladder'], 1] = 255   #
    
    image[mask == class_index['intestine'], 1] = 255   # intestine (0,255,255)
    image[mask == class_index['intestine'], 2] = 255   #
    image[mask == class_index['liver'], 0] = 255   # liver (255,0,255)
    image[mask == class_index['hepatic_vessel'], 0] = 255  # hepatic_vessel (255,0,255)
    image[mask == class_index['liver_tumor'], 0] = 255  # liver_tumors (255,0,255)
    image[mask == class_index['hepatic_vessel_tumor'], 0] = 255  # hepatic_vessel_tumors (255,0,255)
    image[mask == class_index['liver'], 2] = 255   # liver (255,0,255)
    image[mask == class_index['hepatic_vessel'], 2] = 255  # hepatic_vessel (255,0,255)
    image[mask == class_index['liver_tumor'], 2] = 255  # liver_tumors (255,0,255)
    image[mask == class_index['hepatic_vessel_tumor'], 2] = 255  # hepatic_vessel_tumors (255,0,255)
    
    image[mask == class_index['stomach'], 0] = 255  # stomach (255,128,128)
    image[mask == class_index['stomach'], 1] = 128 + image[mask == class_index['stomach'], 1] * 0.5   
    image[mask == class_index['stomach'], 2] = 128 + image[mask == class_index['stomach'], 2] * 0.5   
    
    image[mask == class_index['spleen'], 0] = 255
    image[mask == class_index['spleen'], 2] = 64   # spleen (255,0,255)
    
    image[mask == class_index['postcava'], 0] = image[mask == class_index['postcava'], 0] * 0.5   # postcava (0,0,255)
    image[mask == class_index['postcava'], 1] = image[mask == class_index['postcava'], 1] * 0.5
    image[mask == class_index['postcava'], 2] = 255

    image[mask == class_index['celiac_trunk'], 0] = 255 # celiac_trunk (255,12,12)
    image[mask == class_index['celiac_trunk'], 1] = image[mask == class_index['celiac_trunk'], 1] * 0.05
    image[mask == class_index['celiac_trunk'], 2] = image[mask == class_index['celiac_trunk'], 2] * 0.05

    image[mask == class_index['esophagus'], 0] = image[mask == class_index['esophagus'], 0] * 0.1 # esophagus (25,255,25)
    image[mask == class_index['esophagus'], 1] = 255
    image[mask == class_index['esophagus'], 2] = image[mask == class_index['esophagus'], 2] * 0.1
    
    image[mask == class_index['portal_vein_and_splenic_vein'], 0] = 0 + image[mask == class_index['portal_vein_and_splenic_vein'], 0] * 0.5 # portal_vein_and_splenic_vein (128,128,255)
    image[mask == class_index['portal_vein_and_splenic_vein'], 1] = 0 + image[mask == class_index['portal_vein_and_splenic_vein'], 1] * 0.5 #
    image[mask == class_index['portal_vein_and_splenic_vein'], 2] = 255  #
    
    image[mask == class_index['pancreas'], 0] = 102  # pancreas (102,205,170)
    image[mask == class_index['pancreas'], 1] = 205
    image[mask == class_index['pancreas'], 2] = 170  #
    image[mask == class_index['pancreatic_tumor'], 0] = 102  # pancreatic_tumors (102,205,170)
    image[mask == class_index['pancreatic_tumor'], 1] = 205
    image[mask == class_index['pancreatic_tumor'], 2] = 170

    image[mask == class_index['adrenal_gland_right'], 0] = 200 + image[mask == class_index['adrenal_gland_right'], 0] * 0.2  # adrenal_gland_right (200,128,0)
    image[mask == class_index['adrenal_gland_right'], 2] = 128 + image[mask == class_index['adrenal_gland_right'], 2] * 0.5  #
    image[mask == class_index['adrenal_gland_left'], 0] = 200 + image[mask == class_index['adrenal_gland_left'], 0] * 0.2  # lung_left (200,128,0)
    image[mask == class_index['adrenal_gland_left'], 2] = 128 + image[mask == class_index['adrenal_gland_left'], 2] * 0.5  #
    
    image[mask == class_index['duodenum'], 0] = 255 # duodenum (255,80,80)
    image[mask == class_index['duodenum'], 1] = 80 + image[mask == class_index['duodenum'], 1] * 0.6 #
    image[mask == class_index['duodenum'], 2] = 80 + image[mask == class_index['duodenum'], 2] * 0.6 #
    
    image[mask == class_index['lung_right'], 0] = image[mask == class_index['lung_right'], 0] * 0.2  # lung_right (0,0,128)
    image[mask == class_index['lung_right'], 2] = 128 + image[mask == class_index['lung_right'], 2] * 0.5  #
    image[mask == class_index['lung_left'], 0] = image[mask == class_index['lung_left'], 0] * 0.2  # lung_left (0,0,128)
    image[mask == class_index['lung_left'], 2] = 128 + image[mask == class_index['lung_left'], 2] * 0.5  #
    image[mask == class_index['lung_tumor'], 0] = 200 + image[mask == class_index['lung_tumor'], 0] * 0.2  # lung_tumor (200,0,64)
    image[mask == class_index['lung_tumor'], 2] = 64 + image[mask == class_index['lung_tumor'], 2] * 0.5  #
    
    image[mask == class_index['colon'], 0] = 170  # colon (170,0,255)
    image[mask == class_index['colon'], 1] = 0 + image[mask == class_index['colon'], 1] * 0.7    #
    image[mask == class_index['colon'], 2] = 255  #
    image[mask == class_index['colon_tumor'], 0] = 170  # colon_tumors (170,0,255)
    image[mask == class_index['colon_tumor'], 1] = 0 + image[mask == class_index['colon_tumor'], 1] * 0.7    #
    image[mask == class_index['colon_tumor'], 2] = 255  #
    
    image[mask == class_index['prostate'], 0] = 0    # prostate (0,128,128)
    image[mask == class_index['prostate'], 1] = 128  #
    image[mask == class_index['prostate'], 2] = 128 + image[mask == class_index['prostate'], 2] * 0.5  #
    
    image[mask == class_index['rectum'], 0] = 255    # rectum (255,0,0)
    image[mask == class_index['rectum'], 1] = 0 + image[mask == class_index['rectum'], 1] * 0.5  #
    image[mask == class_index['rectum'], 2] = 0  #
    
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
#    print(folder_names)
     
    print('>> {} CPU cores are secured.'.format(cpu_count()))
    
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
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
