import nibabel as nib
import numpy as np
import cv2
import os

# resize from None * None * 35 to None * None * 64
def level_resize(img):
    if img.shape[2] < 70:
        temp = np.zeros((img.shape[0], img.shape[1], 70))

        for i in range(35):
            each_slice = img[:, :, i]
            temp[:, :, 2 * i] = each_slice
            temp[:, :, 2 * i + 1] = each_slice
        
        result_img = temp[:, :, 3:67]
    else:
        result_img = img[:, :, 3:67]
    
    return result_img

def resize(img):
    target_size = 144
    result_img = np.zeros((target_size, target_size, img.shape[2]))

    for i in range(img.shape[2]):
        each_slice = img[:, :, i]
        resized_slice = cv2.resize(each_slice, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        result_img[:, :, i] = resized_slice
    
    result_img = level_resize(result_img)

    return result_img

FLAIR_PATH = './other_data/flairs/'
flair_list = os.listdir(FLAIR_PATH)
MASK_PATH = './other_data/masks/'
mask_list = os.listdir(MASK_PATH)

flairs = []
masks = []

for flair in flair_list:
    img = nib.load(FLAIR_PATH + flair)
    img = img.get_fdata()
    img = resize(img)
    
    flairs.append(img)

for mask in mask_list:
    img = nib.load(MASK_PATH + mask)
    img = img.get_fdata()
    img = resize(img)
    
    for i in range(4):
        masks.append(img)

flairs = np.array(flairs)
masks = np.array(masks)

print(flairs.shape)
print(masks.shape)

np.save('./dataset/other_mris.npy', flairs)
np.save('./dataset/other_4_masks.npy', masks)