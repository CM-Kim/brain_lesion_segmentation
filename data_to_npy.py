import nibabel as nib
import numpy as np
import glob
import re
import cv2

# resize to 128 * 128 * 48
def resize(img):
    target_size = 128
    result_img = np.zeros((target_size, target_size, 48))

    for i in range(48):
        each_slice = img[:, :, i]
        resized_slice = cv2.resize(each_slice, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        result_img[:, :, i] = resized_slice
    
    return result_img

# file search
t1s = glob.glob('*/*/*T1_to_MAG.nii.gz')
t2s = glob.glob('*/*/*T2_to_MAG.nii.gz')
flair = glob.glob('*/*/*T2FLAIR_to_MAG.nii.gz')
seg = glob.glob('*/*/*T2FLAIR_to_MAG_ROI.nii.gz')

t1_img = []
t2_img = []
t2f_img = []
masks = []

pat = re.compile('.*\.nii\.gz')

# all file resize and append to arr
for items in list(zip(t1s, t2s, flair, seg)):
    for item in items:
        print(item)
        img = nib.load(item)
        img = img.get_fdata()
        
        if(img.shape[2] != 48):
            over = int((img.shape[2] - 48) / 2)
            img = img[:, :, over : img.shape[2] - over]
            
        img = resize(img)

        if item.endswith('FLAIR_to_MAG.nii.gz'):
            t2f_img.append(img)
        elif item.endswith('T1_to_MAG.nii.gz'):
            t1_img.append(img)
        elif item.endswith('T2_to_MAG.nii.gz'):
            t2_img.append(img)
        else:
            masks.append(img)


t1_img = np.array(t1_img)
t2_img = np.array(t2_img)        
t2f_img = np.array(t2f_img)
masks = np.array(masks)

print(t1_img.shape)
print(t2_img.shape)
print(t2f_img.shape)
print(masks.shape)

# save fil
np.save('./dataset/t1s.npy', t1_img)
np.save('./dataset/t2s.npy', t2_img)
np.save('./dataset/flairs.npy', t2f_img)
np.save('./dataset/masks.npy', masks)