'''
python -W ignore plot_video_multiprocessing.py --abdomen_atlas /Users/zongwei.zhou/Dropbox\ \(ASU\)/PublicResource/SuPreM/AbdomenAtlas/AbdomenAtlas1.0 --png_save_path /Users/zongwei.zhou/Desktop/AbdomenAtlas1.0PNG --video_save_path /Users/zongwei.zhou/Desktop/AbdomenAtlas1.0AVI --gif_save_path /Users/zongwei.zhou/Desktop/AbdomenAtlas1.0GIF
'''

import numpy as np 
import os 
import cv2
import argparse
import nibabel as nib 
from tqdm import tqdm 
from PIL import Image
import imageio
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

low_range = -150
high_range = 250

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
    'portal and splenic vein': 10,
    'pancreas': 11,
    'adrenal gland R': 12,
    'adrenal gland L': 13,
    'duodenum': 14,
    'hepatic vessel': 15,
    'lung R': 16,
    'lung L': 17,
    'colon': 18,
    'intestine': 19,
    'rectum': 20,
    'bladder': 21,
    'prostate': 22,
    'head of femur L': 23,
    'head of femur R': 24,
    'celiac trunk': 25,
    'kidney tumor': 26,
    'liver tumor': 27,
    'pancreatic tumor': 28,
    'hepatic vessel tumor': 29,
    'lung tumor': 30,
    'colon tumor': 31,
    'kidney cyst': 32,
}

def add_colorful_mask(image, mask, class_index):
    
    image[mask == class_index['spleen'], 0] = 255   # spleen (255,0,0) 
    
    image[mask == class_index['kidney_right'], 1] = 255   # kidney_right (0,255,0)
    image[mask == class_index['kidney_left'], 1] = 255   # kidney_left (0,255,0)
    image[mask == class_index['kidney tumor'], 1] = 255  # kidney tumor (0,255,0)
    image[mask == class_index['kidney cyst'], 1] = 255  # kidney cyst (0,255,0)
    
    image[mask == class_index['gall_bladder'], 0] = 255   # gall_bladder (255,255,0)
    image[mask == class_index['gall_bladder'], 1] = 255   # 
    
    # image[mask == class_index['esophagus'], 1] = 255   # esophagus (0,255,255)
    # image[mask == class_index['esophagus'], 2] = 255   # 
    image[mask == class_index['liver'], 0] = 255   # liver (255,0,255)
    image[mask == class_index['hepatic vessel'], 0] = 255  # hepatic vessel (255,0,255)
    image[mask == class_index['liver tumor'], 0] = 255  # liver tumors (255,0,255)
    image[mask == class_index['hepatic vessel tumor'], 0] = 255  # hepatic vessel tumors (255,0,255)
    image[mask == class_index['liver'], 2] = 255   # liver (255,0,255)
    image[mask == class_index['hepatic vessel'], 2] = 255  # hepatic vessel (255,0,255)
    image[mask == class_index['liver tumor'], 2] = 255  # liver tumors (255,0,255)
    image[mask == class_index['hepatic vessel tumor'], 2] = 255  # hepatic vessel tumors (255,0,255)
    
    image[mask == class_index['stomach'], 0] = 255
    image[mask == class_index['stomach'], 1] = 239   # stomach (255,239,255)
    image[mask == class_index['stomach'], 2] = 213   # 
    
    image[mask == class_index['aorta'], 1] = 255
    image[mask == class_index['aorta'], 2] = 255   # aorta (0,255,255)
    
    image[mask == class_index['postcava'], 0] = 205   # postcava (205,133,63)
    image[mask == class_index['postcava'], 1] = 133   # 
    image[mask == class_index['postcava'], 2] = 63 # + image[mask == class_index['postcava'], 2] * 0.2   # 
    
    # image[mask == class_index['portal and splenic vein'], 0] = 0 + image[mask == class_index['portal and splenic vein'], 0] * 0.5 # portal and splenic vein (0,0,255)
    # image[mask == class_index['portal and splenic vein'], 1] = 0 + image[mask == class_index['portal and splenic vein'], 1] * 0.5 # 
    # image[mask == class_index['portal and splenic vein'], 2] = 255  # 
    
    image[mask == class_index['pancreas'], 0] = 102  # pancreas (102,205,170)
    image[mask == class_index['pancreas'], 1] = 205
    image[mask == class_index['pancreas'], 2] = 170  #  
    image[mask == class_index['pancreatic tumor'], 0] = 102  # pancreatic tumors (102,205,170)
    image[mask == class_index['pancreatic tumor'], 1] = 205
    image[mask == class_index['pancreatic tumor'], 2] = 170

    # image[mask == class_index['adrenal gland R'], 0] = 0 + image[mask == class_index['adrenal gland R'], 0] * 0.5 # adrenal gland R (0,255,0)
    # image[mask == class_index['adrenal gland R'], 1] = 255 # 
    # image[mask == class_index['adrenal gland R'], 2] = 0 + image[mask == class_index['adrenal gland R'], 2] * 0.5  # 
    # image[mask == class_index['adrenal gland L'], 0] = 0 + image[mask == class_index['adrenal gland L'], 0] * 0.5 # adrenal gland L (0,255,0)
    # image[mask == class_index['adrenal gland L'], 1] = 255 # 
    # image[mask == class_index['adrenal gland L'], 2] = 0 + image[mask == class_index['adrenal gland L'], 2] * 0.5 # 
    
    # image[mask == class_index['duodenum'], 0] = 255 # duodenum (255,80,80)
    # image[mask == class_index['duodenum'], 1] = 80 + image[mask == class_index['duodenum'], 1] * 0.6 # 
    # image[mask == class_index['duodenum'], 2] = 80 + image[mask == class_index['duodenum'], 2] * 0.6 # 
    
    # image[mask == class_index['lung R'], 0] = 200 + image[mask == class_index['lung R'], 0] * 0.2  # lung R (200,128,0)
    # image[mask == class_index['lung R'], 2] = 128 + image[mask == class_index['lung R'], 2] * 0.5  #
    # image[mask == class_index['lung L'], 0] = 200 + image[mask == class_index['lung L'], 0] * 0.2  # lung L (200,128,0)
    # image[mask == class_index['lung L'], 2] = 128 + image[mask == class_index['lung L'], 2] * 0.5  #
    # image[mask == class_index['lung tumor'], 0] = 200 + image[mask == class_index['lung tumor'], 0] * 0.2  # lung tumor (200,128,0)
    # image[mask == class_index['lung tumor'], 2] = 128 + image[mask == class_index['lung tumor'], 2] * 0.5  #
    
    # image[mask == class_index['colon'], 0] = 170  # colon (170,0,255)
    # image[mask == class_index['colon'], 1] = 0 + image[mask == class_index['colon'], 1] * 0.7    # 
    # image[mask == class_index['colon'], 2] = 255  # 
    # image[mask == class_index['colon tumor'], 0] = 170  # colon tumors (170,0,255)
    # image[mask == class_index['colon tumor'], 1] = 0 + image[mask == class_index['colon tumor'], 1] * 0.7    # 
    # image[mask == class_index['colon tumor'], 2] = 255  #
    
    # image[mask == class_index['prostate'], 0] = 0    # prostate (0,128,128)
    # image[mask == class_index['prostate'], 1] = 128  # 
    # image[mask == class_index['prostate'], 2] = 128 + image[mask == class_index['prostate'], 2] * 0.5  # 
    
    # image[mask == class_index['celiac trunk'], 0] = 255    # celiac trunk (255,0,0)
    # image[mask == class_index['celiac trunk'], 1] = 0 + image[mask == class_index['celiac trunk'], 1] * 0.5  # 
    # image[mask == class_index['celiac trunk'], 2] = 0  # 
    
    return image

def load_individual_maps(segmentation_dir):
    
    mask_path = os.path.join(segmentation_dir, 'liver.nii.gz')
    mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    
    for c in class_of_interest:
        mask_path = os.path.join(segmentation_dir, c+'.nii.gz')
        c_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        mask[c_mask == 1] = CLASS_IND[c]
    
    return mask
    
def full_make_png(case_name, args):
    
    for plane in ['axial', 'coronal', 'sagittal']:
        if not os.path.exists(os.path.join(args.png_save_path, plane, case_name)):
            os.makedirs(os.path.join(args.png_save_path, plane, case_name))

    image_name = f'ct.nii.gz'

    image_path = os.path.join(args.abdomen_atlas, case_name, image_name)

    # single case
    image = nib.load(image_path).get_fdata().astype(np.int16)
    mask = load_individual_maps(os.path.join(args.abdomen_atlas, case_name, 'segmentations'))
    
    # change orientation
    ornt = nib.orientations.axcodes2ornt(('F', 'L', 'U'), (('L','R'),('B','F'),('D','U')))
    image = nib.orientations.apply_orientation(image, ornt)
    mask = nib.orientations.apply_orientation(mask, ornt)
    
    image[image > high_range] = high_range
    image[image < low_range] = low_range
    image = np.round((image - low_range) / (high_range - low_range) * 255.0).astype(np.uint8)
    image = np.repeat(image.reshape(image.shape[0],image.shape[1],image.shape[2],1), 3, axis=3)
    
    image_mask = add_colorful_mask(image, mask, CLASS_IND)
    
    for z in range(mask.shape[2]):
        Image.fromarray(image_mask[:,:,z,:]).save(os.path.join(args.png_save_path, 'axial', case_name, str(z)+'.png'))

    for z in range(mask.shape[1]):
        Image.fromarray(image_mask[:,z,:,:]).save(os.path.join(args.png_save_path, 'sagittal', case_name, str(z)+'.png'))

    for z in range(mask.shape[0]):
        Image.fromarray(image_mask[z,:,:,:]).save(os.path.join(args.png_save_path, 'coronal', case_name, str(z)+'.png'))
        
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
     folder_names = [name for name in os.listdir(args.abdomen_atlas) if os.path.isdir(os.path.join(args.abdomen_atlas, name))]
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
    parser.add_argument("--FPS", dest='FPS', type=float, default=20,
                        help='the FPS value for videos',
                       )
    args = parser.parse_args()

    if not os.path.exists(args.png_save_path):
        os.makedirs(args.png_save_path)

    if not os.path.exists(args.video_save_path):
        os.makedirs(args.video_save_path)

    if not os.path.exists(args.gif_save_path):
        os.makedirs(args.gif_save_path)
    
    main(args)
