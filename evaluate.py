import keras
import matplotlib.pyplot as plt
import numpy as np
from intel_unet import unet

unet_model = unet()

# load trained model
model = keras.models.load_model('best_checkpoint.h5', custom_objects={
    'dice_coef' : unet_model.dice_coef,
    'soft_dice_coef' : unet_model.soft_dice_coef,
    'dice_coef_loss' : unet_model.dice_coef_loss,
    'combined_dice_ce_loss' : unet_model.combined_dice_ce_loss,
    'sensitivity' : unet_model.sensitivity,
    'specificity' : unet_model.specificity
    })

# load data
PATH = './dataset/'

images = np.load(PATH + 'flairs.npy')
masks = np.load(PATH + 'masks.npy')

images = np.expand_dims(images, axis=-1)
masks = np.expand_dims(masks, axis=-1)

# predict
predict = (model.predict(images) > 0.2)*1

# show result
plt.figure(figsize=(8,20))
i=1;total=5
temp = np.ones_like( masks[0, :, :, 32] )
for idx in np.random.randint(0,high=images.shape[0],size=total):
    plt.subplot(total,3,i);i+=1
    plt.imshow( np.squeeze(images[idx, :, :, 32],axis=-1), cmap='gray' )
    plt.title("MRI Image");plt.axis('off')
    
    plt.subplot(total,3,i);i+=1
    plt.imshow( np.squeeze(images[idx, :, :, 32],axis=-1), cmap='gray' )
    plt.imshow( np.squeeze(masks[idx, :, :, 32],axis=-1), alpha=0.8, cmap='Reds' )
    plt.title("Original Mask");plt.axis('off')
    
    plt.subplot(total,3,i);i+=1
    plt.imshow( np.squeeze(images[idx, :, :, 32],axis=-1), cmap='gray' )
    plt.imshow( np.squeeze(predict[idx, :, :, 32],axis=-1),  alpha=0.8, cmap='Reds' )
    plt.title("Predicted Mask");plt.axis('off')

plt.show()