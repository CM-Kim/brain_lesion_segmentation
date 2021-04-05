import numpy as np

# Data load
PATH = './dataset/'

images = np.load(PATH + 'images.npy')
masks = np.load(PATH + 'masks.npy')

images = np.expand_dims(images, axis=-1)
masks = np.expand_dims(masks, axis=-1)

# train data - validation data split
from sklearn.model_selection import train_test_split
import gc

X, X_v, Y, Y_v = train_test_split(images, masks, test_size=0.2)

del images
del masks

gc.collect()

# train
import keras
from keras.losses import binary_crossentropy
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from intel_unet import unet

# load model and compile
unet_model = unet()
unet_model.model.compile(
    optimizer=unet_model.optimizer,
    loss=unet_model.loss,
    metrics=unet_model.metrics)

# callback setting
checkpoint = keras.callbacks.ModelCheckpoint('./best_checkpoint.h5', verbose=1, save_best_only=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                          patience=5, min_lr=0.0001)

# fit
hist = unet_model.model.fit(X, Y, batch_size=2, epochs=100, validation_data=(X_v, Y_v), verbose=1, callbacks=[checkpoint, reduce_lr])

# print performance graph
import matplotlib.pyplot as plt

f, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 4))
t = f.suptitle('Unet Performance in Segmenting Lesions', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)
epoch_list = hist.epoch

ax2.plot(epoch_list, hist.history['loss'], label='Train Loss')
ax2.plot(epoch_list, hist.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epoch_list[-1], 5))
ax2.set_ylabel('Loss Value');ax2.set_xlabel('Epoch');ax2.set_title('Loss')
ax2.legend(loc="best");ax2.grid(color='gray', linestyle='-', linewidth=0.5)

ax3.plot(epoch_list, hist.history['dice_coef'], label='Train Dice coef')
ax3.plot(epoch_list, hist.history['val_dice_coef'], label='Validation Dice coef')
ax3.set_xticks(np.arange(0, epoch_list[-1], 5))
ax3.set_ylabel('Dice coef');ax3.set_xlabel('Epoch');ax3.set_title('Dice coef')
ax3.legend(loc="best");ax3.grid(color='gray', linestyle='-', linewidth=0.5)

plt.show()