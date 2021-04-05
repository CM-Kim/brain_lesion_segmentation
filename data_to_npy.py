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
flair = glob.glob('*/*/*T2FLAIR_to_MAG.nii.gz')
seg = glob.glob('*/*/*T2FLAIR_to_MAG_ROI.nii.gz')

t2f_img = []
masks = []

pat = re.compile('.*\.nii\.gz')

# all file resize and append to arr
for items in list(zip(flair, seg)):
    for item in items:
        print(item)
        img = nib.load(item)
        img = img.get_fdata()
        
        if(img.shape[2] != 48):
            over = int((img.shape[2] - 48) / 2)
            img = img[:, :, over : img.shape[2] - over]

        print(img.shape)
        img = resize(img)
        print(img.shape)
        is_flair = item.endswith('MAG.nii.gz')

        if is_flair:
            t2f_img.append(img)
        else:
            masks.append(img)
        
t2f_img = np.array(t2f_img)
masks = np.array(masks)

print(t2f_img.shape)
print(masks.shape)

# save fil
np.save('./dataset/images_expand.npy', t2f_img)
np.save('./dataset/masks_expand.npy', masks)