import numpy as np
import tensorflow as tf
import time
import sys, os
sys.path.append('/home/hpcgies1/Masters-Thesis/AttemptFour')
#import utils
import tqdm
from nsd_access import NSDAccess
import pandas as pd
import cv2
from DataLoaders import load_avg_betas2 as loader


# GET NSD KEYS
all_nsd_keys = []
for i in range(8):
    nsd_keys, shr_nsd_keys, test_keys = loader.get_nsd_keys(str(i + 1))
    all_nsd_keys.extend(nsd_keys)
print("all_nsd_keys:", len(all_nsd_keys))



# LOAD IMAGE MODEL
image_model = tf.keras.applications.VGG16(include_top = True, weights = 'imagenet')
print("Image model loaded ... ")

#for i in range(len(image_model.layers)):
#    print(i, image_model.layers[i].output)
#print()

new_input = image_model.input
#hidden_layer = image_model.layers[-2].output # take last fc layer (4096,)
hidden_layer = image_model.layers[-6].output # last conv layer
print("output layer shape:", hidden_layer.shape)

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
print("Image model built ... ")



which_split = "Train"



def load_image(img_paths: list):
    """ Given image path - load and pre-process it """
    imgs = []
    for path in img_paths:
        img = cv2.imread(f"/home/hpcgies1/rds/hpc-work/NIC/Data/Images/mscoco_2017/{which_split}/{which_split.lower()}2017/" + path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.image.resize(img, (224, 224))
        img = tf.keras.applications.vgg16.preprocess_input(img)
        imgs.append(img)
    return np.array(imgs)


## GET THE IMAGE FILE PATHS
image_paths = os.listdir(f"/home/hpcgies1/rds/hpc-work/NIC/Data/Images/mscoco_2017/{which_split}/{which_split.lower()}2017/")
print("image_paths:", len(image_paths))
features = np.zeros((len(image_paths), 14, 14, 512), dtype=np.float32)
print("features:", features.shape)


## Map image file to its index in the output
image_path_to_index = {}
for k,v in enumerate(image_paths):
    image_path_to_index[ int(v.split(".")[0]) ] = k

image_path_index_df = pd.DataFrame.from_dict(image_path_to_index, orient='index', columns=['idx'])
image_path_index_df.index.name = 'cocoId'
image_path_index_df.to_csv(f"/home/hpcgies1/rds/hpc-work/NIC/Data/Images/mscoco_2017/{which_split.lower()}_img_id_to_idx.csv")





## RUN THE MODEL

batch_size = 128
batch_count = 0
for i in tqdm.tqdm(range(0, features.shape[0], batch_size)):
    start_t = time.time()
    file_path = image_paths[i:i+batch_size]
    imgs = load_image(file_path)
    features[i:i+batch_size, :, :] = image_features_extract_model(imgs)
    batch_count += 1
    print(f"Batch {batch_count} // {(time.time() - start_t):.2f} sec")

print("features:", features.shape)
features = np.reshape(features, (features.shape[0], 196, 512))
print("reshaped features:", features.shape)

with open(f"/home/hpcgies1/rds/hpc-work/NIC/Data/Images/mscoco_2017/{which_split.lower()}_vgg16.npy", "wb") as f:
    np.save(f, features)
print("training set saved ")



print("done.")
