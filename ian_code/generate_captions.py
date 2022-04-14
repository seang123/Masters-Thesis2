import numpy as np
import os, sys
import pandas as pd
from nsd_access import NSDAccess
from collections import defaultdict
import json

"""
subject 02 has 34 trials that have 6 captions the rest have 5
"""

subject_n = '2'
subject = f'subj_{subject_n}'

nsda = NSDAccess("/home/hpcgies1/rds/hpc-work/NIC/NSD")

#loc = f"/fast/seagie/data/captions"
loc = f"/home/hpcgies1/rds/hpc-work/NIC/Data/captions"

if not os.path.exists(loc):
    print(f"creating dir: {loc}")
    os.makedirs(loc)
else:
    print(f"dir {loc} already exists")

df = pd.read_csv(f"./TrainData/subj0{subject_n}_conditions.csv")

print(df.columns)
print(df.head(1))

shrd = df['nsd_key'].loc[df['is_shared']==1] - 1
unq = df['nsd_key'].loc[df['is_shared']==0] - 1

print("shrd", len(shrd))
print("unq", len(unq))

### The NSD keys are index 1 .. 73000 but read_image_coco_info uses 0 indexing so everything is off by one with respect to the brain betas?
#captions = nsda.read_image_coco_info(image_index=[0])
#print(captions)
try:
    captions = nsda.read_image_coco_info(image_index=[73000])
    print(captions)
except Exception:
    print("73000 as index doesn't work since we index from 0 so 72999 is the last index")


all_keys = list(df['nsd_key'].values - 1)
all_keys = list(range(0, 73000))

captions = [i for i in nsda.read_image_coco_info(image_index=all_keys)] # [[{}]]
captions_data = []
for k, v in enumerate(all_keys):
    cur_captions = []
    cap = [j['caption'] for j in captions[k]]
    cap = cap[:5]
    assert len(cap) == 5
    """
    with open(f'{loc}/KID{v+1}.txt', "w") as f:
        for c in cap:
            c = c.replace("\n", " ")
            f.write(f"{c}\n")
    """
    for c in cap:
        c = c.replace("\n", " ")
        cur_captions.append(c)
    captions_data.append(cur_captions)
    print(f"batch: {k}", end='\r')

print()
print("len(capitons_data)", len(captions_data))
for i in range(len(captions_data)):
    if len(captions_data[i]) != 5:
        print(captions_data[i])
        raise

with open(f'{loc}/all_captions.txt', "w") as f:
    for k, v in enumerate(all_keys):
        for i in range(5):
            f.write(f'{v+1}\t{captions_data[k][i]}\n')
    print("All captions saved to single file: all_captions.txt")



