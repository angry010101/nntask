# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import itertools
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy
import numpy as np
import rasterio
import rasterio.mask
import tensorflow
from keras import layers, Model
from keras.callbacks import ModelCheckpoint
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from keras.utils import array_to_img
from numpy import array
from rasterio.features import rasterize
from rasterio.plot import reshape_as_image
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from tensorflow import keras, newaxis
from tensorflow.python.keras.backend import argmax
from tensorflow.python.ops.gen_math_ops import log
from torch.utils.data import Dataset

import weighted_sparse_categorical_crossentropy

raster_path = "T36UXV_20200406T083559_TCI_10m.jp2"

from matplotlib.colors import LinearSegmentedColormap

ncolors = 256
color_array = plt.get_cmap('gist_rainbow')(range(ncolors))

# change alpha values
color_array[:,-1] = np.linspace(1.0,0.0,ncolors)

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)

# register this new colormap with matplotlib
plt.register_cmap(cmap=map_object)


def poly_from_utm(polygon, transform):
    poly_pts = []

    # make a polygon from multipolygon
    poly = cascaded_union(polygon)
    for i in np.array(poly.exterior.coords):
        # transfrom polygon to image crs, using raster meta
        poly_pts.append(~transform * tuple(i))

    # make a shapely Polygon object
    new_poly = Polygon(poly_pts)
    return new_poly


checkpoint_filepath = 'checkpoint/'
mask = "masks/Masks_T36UXV_20190427.shp"
train_df = gpd.read_file(mask)


def generateTrainingSet():
    with rasterio.open(raster_path, "r", driver="JP2OpenJPEG") as src:
        raster_img = src.read()
        raster_meta = src.meta
    src = rasterio.open(raster_path, 'r')
    failed = []
    mask = "masks/Masks_T36UXV_20190427.shp"

    train_df = gpd.read_file(mask)

    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    # let's remove rows without geometry
    train_df = train_df[train_df.geometry.notnull()]

    # assigning crs
    train_df.crs = {'init': 'epsg:4267'}

    # transforming polygons to the raster crs
    train_df = train_df.to_crs({'init': raster_meta['crs']['init']})

    for num, row in train_df.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = poly_from_utm(p, src.meta['transform'])
                poly_shp.append(poly)

    mask = rasterize(shapes=poly_shp,
                     out_shape=im_size)

    plt.figure(figsize=(15, 15))
    return mask


def pixel_acc(inputs, targs):
    inputs = inputs.argmax(dim=1)[:, None, ...]
    return (targs[targs != 0] == inputs[targs != 0]).float().mean()


# inv_freq = np.array(1/(train_df.value_counts()))

# inv_freq = [0.,*inv_freq]

# inv_prop = tensor(inv_freq/sum(inv_freq)).float().cuda()


def getMap():
    raster_path = "T36UXV_20200406T083559_TCI_10m.jp2"
    with rasterio.open(raster_path, "r", driver="JP2OpenJPEG") as src:
        raster_img = src.read()
        raster_meta = src.meta

    print(raster_img.shape)
    raster_img = reshape_as_image(raster_img)

    return raster_img


def verifyX(X):
    _, axs = plt.subplots(4, 4, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(X, axs):
        ax.imshow(img)
    plt.show()


from sklearn.model_selection import train_test_split

def double_conv_block(x, n_filters):

   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)

   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x

def display(display_list):
 plt.figure(figsize=(15, 15))
 title = ["Input Image", "True Mask", "Predicted Mask"]
 for i in range(len(display_list)):
   plt.subplot(1, len(display_list), i+1)
   plt.title(title[i])
   plt.imshow(keras.utils.array_to_img(display_list[i]))
   plt.axis("off")
 plt.show()


from keras.layers import *


def unetmodel():
    inputs = layers.Input(shape=(128, 128, 3))
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)


from scipy.ndimage.interpolation import rotate
def createData(source, map):
    w = map.shape[0]
    stepw = 128  # int(w / grids)
    h = map.shape[1]
    steph = 128  # int(h / grids)
    X = list((map[i:i + stepw, j:j + steph] if (i + stepw < w and j + steph < h) else None) for i, j in
             itertools.product(range(0, map.shape[0], stepw), range(0, map.shape[1], steph)))
    X1 = list(filter(lambda i: i is not None, X))
    Y = list(source[i:i + stepw, j:j + steph] if (i + stepw < w and j + steph < h) else None for i, j in
             itertools.product(range(0, source.shape[0], stepw), range(0, source.shape[1], steph)))
    Y = list(filter(lambda i: i is not None, Y))
    Y1 = np.expand_dims(Y, axis=3)
    X = []
    Y = []
    empty_terrain_limit = 0 #len(X1)/1000
    print(f"EMPTY TERRAINS {empty_terrain_limit}")
    for i in range(0, len(X1)):
        x = X1[i]
        y = Y1[i]
        if np.count_nonzero(y) > 0:
            X.append(x)
            Y.append(y)
            for angle in (90, 180, 270):
                #augmentation
                X.append(rotate(x, angle=angle))
                Y.append(rotate(y, angle=angle))
        else:
            if empty_terrain_limit > 0:
                empty_terrain_limit -= 1
                X.append(x)
                Y.append(y)

    print(f"DATA LEN {len(X)}")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=41)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, random_state=42, test_size=0.5)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val


class CustomImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return np.array(image.resize((224, 224), resample=0)), label



import segmentation_models as sm


def create_mask(pred_mask):
    pred_mask = argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., newaxis]
    return pred_mask[0]


BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

mapp = map

def add_sample_weights(label):
  # The weights for each class, with the constraint that:
  #     sum(class_weights) == 1.0
  class_weights = tensorflow.constant([100000000, 0])
  class_weights = class_weights/tensorflow.reduce_sum(class_weights)

  # Create an image of `sample_weights` by using the label at each pixel as an
  # index into the `class weights` .
  sample_weights = tensorflow.gather(class_weights, indices=tensorflow.cast(label, tensorflow.int32))

  return sample_weights

def nnprogram(X_train, Y_train, X_test, Y_test, X_val, Y_val):
    X = X_train
    Y = Y_train
    ROWS = X[0].shape[0]
    COLS = X[0].shape[1]
    CHANNELS = X[0].shape[2]
    model = unetmodel()
    nonzeroys = np.count_nonzero(Y_train)
    sample_weight = np.ones(shape=(len(Y_train), 128, 128, 1))
    cl1 = ((len(Y_train)*128**2-nonzeroys)/len(Y_train)*128**2)
    cl2 = (nonzeroys/len(Y_train)*128**2)
    k = 600
    for i, sample in enumerate(Y_train):
        sample_weight[i] = sample_weight[i]+sample*k

    def jaccard_distance_loss(y_true, y_pred, smooth=100):
        K = keras.backend
        y_true = keras.backend.cast(y_true, "float32")
        """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

        The jaccard distance loss is usefull for unbalanced datasets. This has been
        shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
        gradient.

        Ref: https://en.wikipedia.org/wiki/Jaccard_index

        @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
        @author: wassname
        """
        intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
        sum_ = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth

    def iou_loss(y_true, y_pred):
        return jaccard_distance_loss(y_true, y_pred)
    total_loss = iou_loss

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='max',
        save_best_only=True)

    optimizer = Adam(lr=0.001)

    model.compile(
        optimizer=optimizer,
        loss=total_loss,
        metrics=[keras.metrics.MeanIoU(num_classes=2)],
    )

    x_train = X
    x_val = X_val
    x_test = X_test

    print(f"WEIGHTS: {k*(cl1/cl2)} {1}")
    model.fit(
        x=array(x_train),
        y=array(Y),
        epochs=4,
        batch_size=12,
        sample_weight=sample_weight,
        validation_data=(array(x_val), array(Y_val)),
        callbacks=[model_checkpoint_callback]
    )

    pred = model.predict(array(x_test))
    for i, t in enumerate(zip(x_test, Y_test, pred)):
        x, y, p = t
        if np.count_nonzero(y) > 0 or np.count_nonzero(p) > 0:
            p = np.rint(p)
            display([x, y, p])
    return 0

if __name__ == '__main__':
    source = generateTrainingSet()
    map = getMap()
    X_train, Y_train, X_test, Y_test, X_val, Y_val = createData(source, map)
    nnprogram(X_train, Y_train, X_test, Y_test, X_val, Y_val)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
