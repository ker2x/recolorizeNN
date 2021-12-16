# %%
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import rgb2lab, lab2rgb, rgb2ycbcr, ycbcr2rgb

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

datadir_color = pathlib.Path("/Volumes/T7/coco/recolorize/color")
datadir_bw = pathlib.Path("/Volumes/T7/coco/recolorize/bw")

# List image in directory
def list_image(path: pathlib.Path):
    imglist = path.glob("*.jpg")
    return [str(img_path) for img_path in imglist]


EPOCHS = 64
LR = 0.0002
BATCH_SIZE = 2

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='swish', padding='same', strides=1, data_format="channels_last",
                                 input_shape=(128, 128, 3)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='swish', padding='same', strides=1))
model.add(tf.keras.layers.MaxPool2D(2,2)) # divide the image by 2
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='swish', padding='same', strides=1))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='swish', padding='same', strides=1))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='swish', padding='same', strides=1))

model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=1, activation='swish', padding='same'))
model.add(tf.keras.layers.Dropout(0.1)) # randomly drop some neurons
model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=1, activation='swish', padding='same'))
model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=1, activation='swish', padding='same'))
model.add(tf.keras.layers.UpSampling2D((2,2))) # rescale x2
model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=1, activation='swish', padding='same'))
model.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=1, activation='swish', padding='same'))

model.add(tf.keras.layers.Conv2D(3, kernel_size=(3, 3), activation='relu', padding='same', data_format="channels_last"))

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              #              optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0),
              metrics=["accuracy", "mae", "mse"]
              )

color_list = [str(img_path) for img_path in datadir_color.glob("0*.jpg")]
bw_list = [str(img_path) for img_path in datadir_bw.glob("0*.jpg")]

train_x = []
train_y = []

for idx in range(200):
    file_bw = tf.io.read_file(bw_list[idx])
    image_bw = tf.image.decode_jpeg(file_bw, channels=1)
#    image_bw = tf.image.resize_with_crop_or_pad(image=image_bw, target_height=128, target_width=128)
    image_bw = tf.image.resize(image_bw, (128,128))
    image_bw = tf.broadcast_to(image_bw, [128, 128, 3])
    image_bw = tf.cast(image_bw,tf.uint8)
    image_bw = rgb2ycbcr(image_bw)
    train_x.append(image_bw)

    file_color = tf.io.read_file(color_list[idx])
    image_color = tf.image.decode_jpeg(file_color, channels=3)
#    image_color = tf.image.resize_with_crop_or_pad(image=image_color, target_height=128, target_width=128)
    image_color = tf.image.resize(image_color, (128,128))
    image_color = tf.cast(image_color,tf.uint8)
    image_color = rgb2ycbcr(image_color)
    train_y.append(image_color)

train_y = np.array(train_y)
train_x = np.array(train_x)

# sequence doesn't works here
# class ImageSequence(tf.keras.utils.Sequence):
#     def __init__(self, color_path, bw_path):
#         self.color_list = [str(img_path) for img_path in color_path.glob("0*.jpg")]
#         self.bw_list = [str(img_path) for img_path in bw_path.glob("0*.jpg")]
#     def __len__(self):
#         #return len(self.color_list)
#         return 500
#     def __getitem__(self, idx):
#         # load
#         file_bw = tf.io.read_file(self.bw_list[idx])
#         file_color = tf.io.read_file(self.color_list[idx])
#         # decode
#         image_bw = tf.image.decode_jpeg(file_bw, channels=1)
#         image_color = tf.image.decode_jpeg(file_color, channels=3)
#         # optional resize
#         image_bw = tf.image.resize_with_crop_or_pad(image=image_bw, target_height=128, target_width=128)
#         image_color = tf.image.resize_with_crop_or_pad(image=image_color, target_height=128, target_width=128)
#         image_bw2 = tf.cast(tf.broadcast_to(image_bw, [128, 128, 3]), tf.float32)
#         return image_bw2, image_color

history = model.fit(train_x, train_y, batch_size=BATCH_SIZE,
                    shuffle=True, epochs=EPOCHS, validation_split=0.1)
# sequence = ImageSequence(datadir_color, datadir_bw)
# history = model.fit(sequence ,epochs=EPOCHS)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


def plot_loss(history):
    plt.figure(1)
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.plot(history.history['loss'], label='training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim((0, 2000))
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss(history)

model.summary()

# get input image
img = train_x[:1]

# predicted
plt.figure(2)
plt.axis("off")
predictions = model.predict(img)
predictions = ycbcr2rgb(predictions)
plt.imshow(tf.squeeze(predictions))
plt.show()

# original colored
# plt.figure(3)
# plt.axis("off")
# original = ycbcr2rgb(train_y[:1])
# plt.imshow(tf.squeeze(original))
# plt.show()

#plt.figure(4)
#original = tf.cast(original,tf.double)
#predictions = tf.cast(predictions,tf.double)
#ssim = tf.image.ssim(original, predictions, 255)
#plt.imshow(ssim)
#plt.show()
