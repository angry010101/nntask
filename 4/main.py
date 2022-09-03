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

def unetmodel():
    inputs = layers.Input(shape=(128, 128, 3))
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # outputs
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u9)
    # unet model with Keras Functional API
    unet_model = Model(inputs, outputs, name="U-Net")
    return unet_model

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)


def createData(source, map):
    grids = 4
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
    empty_terrain_limit = len(X1)/1000
    print(f"EMPTY TERRAINS {empty_terrain_limit}")
    for i in range(0, len(X1)):
        x = X1[i]
        y = Y1[i]
        if np.count_nonzero(y) > 0:
            X.append(x)
            Y.append(y)
        else:
            if empty_terrain_limit > 0:
                empty_terrain_limit -= 1
                X.append(x)
                Y.append(y)

    print(f"DATA LEN {len(X)}")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)
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

# Press the green button in the gutter to run the script.
def calculateError(preds, Y_test, loss):
    print(f"SHAPE {Y_test[0].shape} {preds[0].shape}")
    print(f"Y TEST: {Y_test[0]}")
    preds = np.squeeze(preds)
    print(f"PREDS: {preds[0]}")
    r = loss(np.asarray(Y_test), np.asarray(preds))
    print(f"LOSS {r}")


def nnprogram(X_train, Y_train, X_test, Y_test, X_val, Y_val):
    X = X_train
    Y = Y_train
    ROWS = X[0].shape[0]
    COLS = X[0].shape[1]
    CLASSES = 1
    CHANNELS = X[0].shape[2]
    print(X[0].shape)
    print(f"Y SHAPE: {Y[0].shape}")
    print(f"ROWS {ROWS} {COLS} {CHANNELS}")
    model = unetmodel() #sm.Unet('resnet34', classes=CLASSES, activation='sigmoid', encoder_weights='imagenet')
    #dice_loss = sm.losses.DiceLoss()
    #focal_loss = sm.losses.BinaryFocalLoss()

    nonzeroys = np.count_nonzero(Y_train)
    sample_weight = np.ones(shape=(len(Y_train), 128, 128))
    cl1 = ((len(Y_train)*128**2-nonzeroys)/len(Y_train)*128**2)
    cl2 = (nonzeroys/len(Y_train)*128**2)
    k = 1
    sample_weight[Y_train == 1] = k*cl1/cl2
    sample_weight[Y_train == 0] = 1

    total_loss = "binary_crossentropy",#dice_loss + (1 * focal_loss)

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='max',
        save_best_only=True)

    optimizer = Adam(lr=0.01)

    model.compile(
        optimizer=optimizer,
        loss=total_loss,
        metrics=[keras.metrics.MeanIoU(num_classes=2)],
    )

    x_train = X # preprocess_input(X)
    x_val = X_val # preprocess_input(X_val)
    x_test = X_test # preprocess_input(X_test)



    print(f"WEIGHTS: {k*(cl1/cl2)} {1}")
    model.fit(
            x=array(x_train),
            y=array(Y),
            epochs=20,
            batch_size=24,
            validation_data=(array(x_val), array(Y_val)),
            callbacks=[model_checkpoint_callback]
        )

    for i, t in enumerate(zip(x_test, Y_test)):
        x, y = t
        if np.count_nonzero(y) > 0 or np.count_nonzero(x) > 0:
            pred = model.predict(array(np.expand_dims(x_test[i], axis=0)))
            p = create_mask(pred)
            display([x, y, p])
    return 0

if __name__ == '__main__':
    source = generateTrainingSet()
    map = getMap()
    X_train, Y_train, X_test, Y_test, X_val, Y_val = createData(source, map)
    nnprogram(X_train, Y_train, X_test, Y_test, X_val, Y_val)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
