# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time

start = time.time()

EPOCHS = 64
# LR = 0.0002
LR = 0.0002
BATCH_SIZE = 4
DATASET_SIZE = 128  # Set to 0 for all data

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# disable_eager_execution()
# print(tf.executing_eagerly())

datadir = pathlib.Path("/Volumes/T7/coco/recolorize/color")
ds_list = tf.data.Dataset.list_files(str(datadir / '0*.jpg'), shuffle=True)

# limit the size of the dataset
if DATASET_SIZE != 0:
    ds_list = ds_list.take(DATASET_SIZE)


def process_image(_file_path):
    file = tf.io.read_file(_file_path)
    color = tf.image.decode_jpeg(file, channels=3)
    color = tf.image.random_flip_left_right(color)
    color = tf.image.random_flip_up_down(color)
    #    color = tf.image.resize(color, (128, 128))
    bw = tf.image.rgb_to_grayscale(color)
    bw = tf.broadcast_to(bw, [128, 128, 3])
    bw = tf.image.convert_image_dtype(bw, tf.dtypes.float32)
    color = tf.image.convert_image_dtype(color, tf.dtypes.float32)
    color = tf.image.rgb_to_yiq(color)
    return bw, color


image_ds = ds_list.map(process_image)
# image_ds = image_ds0.cache() # cache in memory (!!!)

image_tensors = image_ds.batch(BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)
ds = image_tensors.prefetch(buffer_size=tf.data.AUTOTUNE)

model = tf.keras.Sequential()
# model.add(tf.keras.layers.DepthwiseConv2D((3, 3), activation='tanh', padding='same', strides=1))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='tanh', padding='same', strides=1, data_format="channels_last",
                                 input_shape=(128, 128, 3)))

model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=1))
model.add(tf.keras.layers.Dropout(0.5))  # randomly drop some neurons
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=1))
model.add(tf.keras.layers.MaxPool2D(2, 2))  # divide the image by 2
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=1))
# model.add(tf.keras.layers.MaxPool2D(2,2)) # divide the image by 2
# model.add(tf.keras.layers.Conv2D(256, (6, 6), activation='swish', padding='same', strides=1))

# model.add(tf.keras.layers.Conv2DTranspose(256, kernel_size=6, strides=1, activation='swish', padding='same'))
model.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(tf.keras.layers.UpSampling2D((2, 2)))  # rescale x2
model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=1, activation='relu', padding='same'))

model.add(tf.keras.layers.Conv2D(3, kernel_size=(3, 3), activation="tanh", padding='same', data_format="channels_last"))

# layer = model.layers
# filters, biases = model.layers[1].get_weights()
# print(layer[1].name, filters.shape)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              metrics=["accuracy", "mae", "mse"]
              )

# history = model.fit(ds, batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS)
history = model.fit(ds, shuffle=False, epochs=EPOCHS)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


def plot_loss(_history):
    plt.figure(1)
    #    plt.plot(history.history['val_loss'], label='validation loss')
    plt.plot(_history.history['loss'], label='training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim((0, 0.01))
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss(history)

# show model
# model.summary()
# plot_model(model, "model.png", show_shapes=True, show_dtype=True)

# get input image
file_path = next(iter(ds_list))
out, _ = process_image(file_path)
image_ds = tf.data.Dataset.from_tensors(out)
img = image_ds.batch(1)

# display original
plt.figure(3)
# out *= 255
plt.imshow(out)
plt.show()

# predicted
plt.figure(2)
plt.axis("off")
predictions = model.predict(img)
predictions = tf.image.yiq_to_rgb(predictions)
# predictions /= 255.0
plt.imshow(tf.squeeze(predictions))
plt.show()

# original colored
# plt.figure(3)
# plt.axis("off")
# original = ycbcr2rgb(train_y[:1])
# plt.imshow(tf.squeeze(original))
# plt.show()

# plt.figure(4)
# original = tf.cast(original,tf.double)
# predictions = tf.cast(predictions,tf.double)
# ssim = tf.image.ssim(original, predictions, 255)
# plt.imshow(ssim)
# plt.show()

print("elapsed : ", time.time() - start)
