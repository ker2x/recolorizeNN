#%%
import tensorflow as tf
import pathlib
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optionally set memory groth to True
# -----------------------------------
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
#%%
datadir_color = pathlib.Path("/Volumes/T7/coco/recolorize/color")
datadir_bw = pathlib.Path("/Volumes/T7/coco/recolorize/bw")

#%%
# List image in directory
def list_image(path: pathlib.Path):
    imglist = path.glob("*.jpg")
    return [str(img_path) for img_path in imglist]

#%%
class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, color_path, bw_path):
        self.color_list = [str(img_path) for img_path in color_path.glob("0*.jpg")]
        self.bw_list = [str(img_path) for img_path in bw_path.glob("0*.jpg")]
    def __len__(self):
        #return len(self.color_list)
        return 50
    def __getitem__(self, idx):
        # load
        file_bw = tf.io.read_file(self.bw_list[idx])
        file_color = tf.io.read_file(self.color_list[idx])
        # decode
        image_bw = tf.image.decode_jpeg(file_bw, channels=1)
        image_color = tf.image.decode_jpeg(file_color, channels=3)
        # optional resize
        image_bw = tf.image.resize_with_crop_or_pad(image=image_bw, target_height=64, target_width=64)
        image_color = tf.image.resize_with_crop_or_pad(image=image_color, target_height=64, target_width=64)
#        print(tf.shape(image_bw))
#        print(tf.shape(image_color))
        return image_bw, image_color

class genImage(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        plt.figure(4)
        plt.clf()
        bw_list = [str(img_path) for img_path in datadir_bw.glob("0*.jpg")]
        file_bw = tf.io.read_file(bw_list[3000])
        image_bw = tf.image.decode_jpeg(file_bw, channels=1)
        predictions = model.predict(image_bw)
        predictions /= 255.0  # tf.cast(predictions_f, tf.int8)
        plt.figure(2)
        plt.imshow(predictions)
        plt.show()


EPOCHS = 16
LR = 0.0005
HIDDENLAYERS = 4
LAYERWIDTH = 9

model = tf.keras.Sequential()

#tf.keras.Input(shape=(1,))
#model.add(tf.keras.layers.Flatten((255,255,3)))
#model.add(tf.keras.layers.Dense(512, activation="swish"))

for _ in range(HIDDENLAYERS):
    model.add(tf.keras.layers.Dense(LAYERWIDTH, activation="swish"))
model.add(tf.keras.layers.Dense(3, activation=None))

model.compile(loss=tf.keras.losses.Huber(delta=1.0),
#              optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0),
              metrics=["accuracy", "mae", "mse"])

sequence = ImageSequence(datadir_color, datadir_bw)

history = model.fit(sequence,epochs=EPOCHS,callbacks=[])

np.set_printoptions(precision=3, suppress=True)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
plt.figure(1)
def plot_loss(history):
#  plt.plot(history.history['val_loss'], label='validation loss')
  plt.plot(history.history['loss'], label='training loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.ylim((0,2))
  plt.legend()
  plt.grid(True)
  plt.show()

plot_loss(history)

bw_list = [str(img_path) for img_path in datadir_bw.glob("0*.jpg")]
file_bw = tf.io.read_file(bw_list[3000])
image_bw = tf.image.decode_jpeg(file_bw, channels=1)
image_bw = tf.image.resize_with_crop_or_pad(image=image_bw, target_height=64, target_width=64)
predictions = model.predict(image_bw)
predictions /= 255.0 #tf.cast(predictions_f, tf.int8)
plt.figure(2)
plt.imshow(image_bw, cmap='gray')
plt.figure(3)
plt.imshow(predictions)
plt.show()
model.summary()
