import numpy as np
import tensorflow as tf
import time
import sys, os
sys.path.append('/home/hpcgies1/Masters-Thesis/AttemptFour')
#import utils
import tqdm
from nsd_access import NSDAccess
import pandas as pd
from DataLoaders import load_avg_betas2 as loader

#gpu_to_use = 0
# Allow memory growth on GPU devices
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#for i in range(0, len(physical_devices)):
#    tf.config.experimental.set_memory_growth(physical_devices[i], True)
#tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')


#### Load subj02 nsd keys
nsd_keys, shr_nsd_keys, test_keys = loader.get_nsd_keys('2')

print("nsd_keys:    ", nsd_keys.shape)
print("shr_nsd_keys:", shr_nsd_keys.shape)

## Subtract 1 from nsd-keys to get 0 idx keys for using with nsd_access
print("nsd keys:", nsd_keys[:10])
nsd_keys     = nsd_keys - 1
shr_nsd_keys = shr_nsd_keys - 1
print("nsd keys:", nsd_keys[:10])

nsd_keys = np.arange(0, 73000) # Get image embeddings for all 73k nsd images

nsd_loader = NSDAccess("/home/hpcgies1/rds/hpc-work/NIC/NSD")
nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)
print("NSD access initalized ...")

image_model = tf.keras.applications.VGG16(include_top = True, weights = 'imagenet')
print("Image model loaded ... ")

for i in range(len(image_model.layers)):
    print(i, image_model.layers[i].output)
print()

new_input = image_model.input
#hidden_layer = image_model.layers[-2].output # take last fc layer (4096,)
hidden_layer = image_model.layers[-6].output # last conv layer

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
print("Image model built ... ")


def load_image(idx: int):
    if not isinstance(idx, list):
        idx = [idx]
    img = nsd_loader.read_images(idx)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img

#train_features = np.zeros((9000, 4096), dtype=np.float32)
#test_features  = np.zeros((1000, 4096), dtype=np.float32)
features = np.zeros((73000, 14, 14, 512), dtype=np.float32)

"""
batch_size = 5
## Train set
for i in tqdm.tqdm(range(0, len(nsd_keys), batch_size)):
    keys = nsd_keys[i:i+batch_size]
    keys = list(keys)
    imgs = load_image(keys)

    feat = image_features_extract_model(imgs) # (bs, 4096)
    train_features[i:i+batch_size] = feat
    #features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))

## Val set
for i in tqdm.tqdm(range(0, len(shr_nsd_keys), batch_size)):
    keys = shr_nsd_keys[i:i+batch_size]
    keys = list(keys)
    imgs = load_image(keys)

    feat = image_features_extract_model(imgs) # (bs, 4096)
    test_features[i:i+batch_size] = feat
    #features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))

print("feature extraction complete ... ")
print(train_features.shape)
print(test_features.shape)

# TODO : save features to .npy file with nsd key as name (make sure to undo -1 key again for name)
print("saving data to disk ... ")
nsd_keys     = nsd_keys + 1
shr_nsd_keys = shr_nsd_keys + 1
"""

batch_size = 32
for i in tqdm.tqdm(range(0, len(nsd_keys), batch_size)):
    keys = nsd_keys[i:i+batch_size]
    keys = list(keys)
    imgs = load_image(keys)
    features[i:i+batch_size, :, :] = image_features_extract_model(imgs)

print("features:", features.shape)
features = np.reshape(features, (features.shape[0], 196, 512))
print("reshaped features:", features.shape)

with open(f"/home/hpcgies1/rds/hpc-work/NIC/Data/Images/mscoco_2017/mscoco_train_vgg16.npy", "wb") as f:
    np.save(f, features)
print("training set saved ")



print("done.")
