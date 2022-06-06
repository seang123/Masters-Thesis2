import os, sys
sys.path.append("/home/hpcgies1/Masters-Thesis/AttemptFour")
import numpy as np
from nsd_access import NSDAccess
import pandas as pd
#import my_utils as uu
import time
import json
import re
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
#from tensorflow.keras.utils import to_categorical
#import collections
from collections import defaultdict
from ian_code import nsd_get_data as ngd
import yaml
import nibabel as nb
from functools import lru_cache
#from concurrent.futures import ThreadPoolExecutor
"""

    Same as load_avg_betas.py but reads froma single .txt file with has all captions in it (in order, 1-73000)

"""

print("------------load_avg_betas.py-------------")

#np.random.seed(42)

thesis_dir = "/home/hpcgies1/Masters-Thesis/AttemptFour/"

nsd_dir = '/home/hpcgies1/rds/hpc-work/NIC/NSD/'
subject = "subj02"
n_sessions = 40
targetspace = 'fsaverage'
captions_path = "/home/hpcgies1/rds/hpc-work/NIC/Data/captions/all_captions.txt"
USE_ENTIRE_CORTEX = True
SEPARATE_HEMISPHERES = True

## ====== Glasser ======
#GLASSER_LH = '/home/danant/misc/lh.HCP_MMP1.mgz'
#GLASSER_RH = '/home/danant/misc/rh.HCP_MMP1.mgz'
GLASSER_LH = "/rds/user/hpcgies1/hpc-work/NIC/NSD/nsddata/freesurfer/fsaverage/label/lh.HCP_MMP1.mgz"
GLASSER_RH = "/rds/user/hpcgies1/hpc-work/NIC/NSD/nsddata/freesurfer/fsaverage/label/rh.HCP_MMP1.mgz"
#VISUAL_MASK = '/home/danant/misc/visual_parcels_glasser.csv'

s = time.time()
glasser_lh = nb.load(GLASSER_LH).get_data() # 163_842 values in the range [0, 180]
glasser_rh = nb.load(GLASSER_RH).get_data()
print(f"load glasser masks: {(time.time() - s):.2f}")

glasser = np.vstack((glasser_lh, glasser_rh)).flatten()

print("glasser_lh", glasser_lh.shape)
print("glasser_rh", glasser_rh.shape)
print("glasser   ", glasser.shape)

#visual_parcels = pd.read_csv(VISUAL_MASK, index_col=0)
#visual_parcel_list = list(visual_parcels.values.flatten())

if USE_ENTIRE_CORTEX == False:
    ## If using only visual cortex
    print("-- visual area glasser regions --")
    groups = []
    glasser_indices = np.array(range(len(glasser))) # [0 to 163841]
    for i in visual_parcel_list:
        group = glasser_indices[glasser==i]
        groups.append(group)
elif SEPARATE_HEMISPHERES:
    print("-- separate hemisphere glasser regions --")
    ## Separate Glasser regions into hemisphers - 360 regions
    glasser_lh = glasser_lh.flatten()
    glasser_rh = glasser_rh.flatten()
    # Right
    glasser_indices_rh = np.array(range(len(glasser_rh))) # [0 to 163_841]
    groups_rh = []
    for i in set(glasser_rh):
        groups_rh.append(glasser_indices_rh[glasser_rh == i])
    # Left
    glasser_indices_lh = np.array(range(len(glasser_lh))) # [0 to 163_841]
    groups_lh = []
    for i in set(glasser_lh):
        groups_lh.append(glasser_indices_lh[glasser_lh == i])
    print("excluding group 0")
    print("groups_lh:", len(groups_lh[1:]))
    print("groups_rh:", len(groups_rh[1:]))
    print("Avg. group size lh:     ", np.mean([len(g) for g in groups_lh[1:]]))
    print("Avg. group size rh:     ", np.mean([len(g) for g in groups_rh[1:]]))
    print("max group size lh | rh: ", np.max([len(g) for g in groups_lh[1:]]), np.max([len(g) for g in groups_rh[1:]]))
    print("min group size lh | rh: ", np.min([len(g) for g in groups_lh[1:]]), np.min([len(g) for g in groups_rh[1:]]))
    groups = groups_lh[1:] + groups_rh[1:]
    print("groups: ", len(groups))
else:
    ## If using entire cortex
    print("-- full cortex glasser regions --")
    groups = []
    glasser_indices = np.array(range(len(glasser)))
    for i in set(glasser):
        group = glasser_indices[glasser == i]
        groups.append(group)
    groups = groups[1:]
    print("sum of groups sizes:", sum([len(g) for g in groups]))
    print("Avg. group size:    ", np.mean([len(g) for g in groups]))
    print("nr of groups        ", len([len(g) for g in groups]))

def get_groups(out_dim, separate_hemi=True):
    return groups, [out_dim for i in range(0,len(groups))]

print("------------load_avg_betas.py-------------")
## =====================

def select_groups(out_dim, remove):
    """
    remove : list(int)
        which Glasser regions to remove by index ([0] would remove only the left hemisphere area V1)
    """
    new_groups = []
    remove = set(remove)
    for i in range(len(groups)):
        if i not in remove:
            new_groups.append(groups[i])
    return new_groups, [out_dim for i in range(0, len(new_groups))]

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        out = func(*args, **kwargs)
        print(f"> {func.__name__} :: {(time.perf_counter() - start):.3f} sec")
        return out
    return wrapper

class Timer():
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, type, value, traceback):
        print(f"{self.name} :: {(time.time() - self.start):.1f}s")


def remove_stop_words(list_of_words: list):
    stop_words = []
    with open(f"stop_words.txt", 'r') as f:
        cont = f.read()
        for i, line in enumerate(cont):
            stop_words.append(str(line.strip()))
    stop_words = set(stop_words)
    return [i for i in list_of_words if i not in stop_words]

@timeit
def load_tokenizer():
    print("Loading tokenizer from disk: fit on all 73k NSD images with vocab size of 5000")
    return tokenizer_from_json(json.load(open(f"{thesis_dir}/TrainData/tokenizer_73k.json", "r")))


@lru_cache(maxsize=None)
def load_all_captions():
    """ Reads captions from text file and returns list of list of captions """
    ls = defaultdict(list)
    with open(captions_path, "r") as f:
        content = f.read().splitlines()
        for k, v in enumerate(content):
            kid, cap = v.split("\t")
            ls[int(kid)].append(cap)
    return ls

def process_caption(cap: str):
    """ pre-process a caption
    by removing some white space, simple punctuation, lower-casing, and adding start/end token
    """
    cap = cap.replace(".", " ").replace(",", " ").strip().split(" ")
    cap = [i.lower() for i in cap if i != '']
    #cap = remove_stop_words(cap)
    cap = ['<start>'] + cap + ['<end>']
    length = len(cap)
    cap = " ".join(cap)
    return cap, length


@timeit
def build_tokenizer(nsd_keys: list, top_k = 5000):
    """
    Build tokenizer from captions

    Ignore captions that aren't in the keys list

    Parameters
    ----------
        nsd_keys : list
            a list of NSD keys to load
        top_k : int
            vocab size

    Returns
    -------
        tokenizer : keras.Tokenizer
        all_captions : list
    """

    all_captions_dict = load_all_captions()
    all_captions_ls = []
    avg_caption_length = 0

    keys = set([i for i in nsd_keys])

    num_files = 0
    for kid, captions in all_captions_dict.items(): #(nsd_key, cap)
        num_files += 1
        for caption in captions:
            cap, length = process_caption(caption)
            avg_caption_length += length
            all_captions_ls.append( cap )

    print(f"num_files scanned: {num_files}")
    print(f"num captions read: {len(all_captions_ls)}")
    print(f"avg caption length: {avg_caption_length/len(all_captions_ls)}")

    tokenizer = Tokenizer(num_words = top_k, oov_token = '<unk>', filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')
    tokenizer.fit_on_texts(all_captions_ls)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    return tokenizer, all_captions_ls

def get_test_set():
    # 515 keys that all 8 subjects saw are our test set
    df = pd.read_csv(f'{thesis_dir}/TrainData/test_conditions.csv')
    return df['nsd_key'].values

def get_nsd_keys(subj: str = '2') -> (list, list):
    """ Get the NSD keys for a subject

    Parameter
    ---------
        subj : int | str
            the subject id
    Returns
    -------
        unq : ndarray
            unique nsd keys
        shr : ndarray
            shared nsd keys
    """

    df = pd.read_csv(f'{thesis_dir}/TrainData/subj0{subj}_conditions2.csv')
    #df_test = pd.read_csv(f'{thesis_dir}/TrainData/test_conditions.csv')

    unq = df['nsd_key'].loc[df['is_shared']==0]
    shrd = df['nsd_key'].loc[df['is_shared']==1]
    #test = df_test['nsd_key'].values
    test = df['nsd_key'].loc[df['is_test']==1].values

    #assert len(unq) == 9000, "incorrect amount of unq keys"
    #assert len(shrd) == 1000, "incorrect amount of shrd keys"
    assert len(test) == 515, f"incorrect amount of test keys: {len(test)}"

    # remove test keys from val set
    shrd = shrd.values
    shrd = np.array([i for i in shrd if i not in test])

    return unq.values, shrd, test


def get_shr_nsd_keys(nsd_dir: str) -> list:
    """ Get the shared NSD keys """
    return ngd.get_1000(nsd_dir)

@timeit
def create_pairs(keys: list, subj='2', single=False):
    """ returns NSD_key - caption pairs

    Parameters
    ----------
        nsd_keys : list
            list of unique nsd keys
        subj: int or str
            an id of the current subject - should match the passed keys
        single: bool
            if True load only the first caption
    Returns
    -------
        pairs : list
            [NSD key, caption, idx(between 0 and 10k), subject id]
    """

    # Loads a dictionary:  {NSD_key -> list(captions)}
    all_captions_dict = load_all_captions()

    pairs = []
    for idx, kid in enumerate(keys):
        captions = all_captions_dict[int(kid)]
        for _, caption in enumerate(captions):
            cap, _ = process_caption(caption)
            pairs.append( (kid, cap, idx, subj) )
            if single:
                break
    return pairs


def batchify(pairs: list, batch_size: int = 64):
    """ Given a list of pairs for each subject (train or val or test) split them into subsets of size batch_size
    with each batch holding data for one subject
    Parameters:
    -----------
        pairs : list
            [[nsd, cap, idx, etc]] : (n_subs, n_samples, 4 - (3, 45000, 4)
        batch_size : int
    Returns:
    --------
        batch_pairs : list
    """

    np.random.shuffle(pairs)

    batches = []
    for p in range(pairs.shape[0]):
        for i in range(0, len(pairs[p]), batch_size):
            batches.append(pairs[p][i:i + batch_size])
    return batches


@timeit
def load_split_betas(subj = '2'):
    """ Load the single betas .npy file and return 3 separate ndarrays for train/val/test
    """
    print(f">> Loading and splitting betas for subject: {subj}")
    df = pd.read_csv(f'{thesis_dir}/TrainData/subj0{subj}_conditions2.csv')
    unq = df['nsd_key'].values
    shrd = df['nsd_key'].loc[df['is_shared']==1].values
    test = df['nsd_key'].loc[df['is_test']==1].values

    # Split the keys into train//val//test
    train_idx = []
    val_idx = []
    test_idx = []
    for i in range(df.shape[0]):
        if df.iloc[i]['is_test'] == 1:
            test_idx.append(i)
        elif df.iloc[i]['is_shared'] == 1:
            val_idx.append(i)
        else:
            train_idx.append(i)

    # List -> np.array
    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    test_idx  = np.array(test_idx)
    print("train_idx:", train_idx.shape)
    print("val_idx:  ", val_idx.shape)
    print("test_idx: ", test_idx.shape)

    # Load betas file
    with Timer('Load betas'):
        betas = np.load(
            open(f"/home/hpcgies1/rds/hpc-work/NIC/Data/subj0{subj}/betas_averaged/all_betas.npy", "rb"))
    # Split betas array
    with Timer('Split betas'):
        train_betas = betas[train_idx, :]
        if val_idx.shape[0] > 0:
            val_betas = betas[val_idx, :]
        else:
            val_betas = np.zeros((1, 1))
        test_betas = betas[test_idx, :]
    return train_betas, val_betas, test_betas


def load_subs(subs = [1, 2, 3, 4, 5, 6, 7, 8]):
    train_pairs = []
    val_pairs = []
    test_pairs = []
    train_betas = {}
    val_betas = {}
    test_betas = {}
    for sub in subs:
        # Load keys and create pairs
        train_keys, val_keys, test_keys = get_nsd_keys(str(sub))
        train_pair = np.array(create_pairs(train_keys, str(sub)))
        val_pair   = np.array(create_pairs(val_keys, str(sub)))
        test_pair  = np.array(create_pairs(test_keys, str(sub), single=True))
        # Store
        train_pairs.append(train_pair)
        val_pairs.append(val_pair)
        test_pairs.append(test_pair)

        # Load betas: {sub: np.array}
        train_beta, val_beta, test_beta = load_split_betas(str(sub))
        train_betas[str(sub)] = train_beta
        val_betas[str(sub)] = val_beta
        test_betas[str(sub)] = test_beta
        print("sub:", sub)
        print("train, val, test:", train_beta.shape, "-", val_beta.shape, "-", test_beta.shape)
        print()

    return train_pairs, val_pairs, test_pairs, train_betas, val_betas, test_betas


def train_val_pairs_from_all(all_pairs):

    train_pairs = []
    val_pairs = []
    for i in range(len(all_pairs)):
        train_pairs.append(all_pairs[i][0])
        val_pairs.append(all_pairs[i][1])
    train_pairs = [i for sublist in train_pairs for i in sublist]
    val_pairs = [i for sublist in val_pairs for i in sublist]

    return train_pairs, val_pairs

def main():

    #train_betas, v_betas, test_betas = load_split_betas()
    #print(train_betas.shape, v_betas.shape, test_betas.shape)

    ls = load_all_captions()
    print(len(ls))

    tok, _ = build_tokenizer(range(1, 73000), top_k=5000)

    train_keys, val_keys, test_keys = get_nsd_keys('2') # (10_000,)
    train_keys_one, _, _ = get_nsd_keys('1') # (10_000,)
    print("len(set(nsd_keys))", len(list(set(train_keys))))
    print("len(set(shr_nsd_keys))", len(list(set(val_keys))))

    train_pairs = create_pairs(train_keys, '2')
    #val_pairs = create_pairs(val_keys, '2')
    #test_pairs = create_pairs(test_keys, '2')
    train_pairs_one = create_pairs(train_keys_one, '1')

    train_pairs = np.array([train_pairs, train_pairs_one])
    print("train_pairs:", train_pairs.shape)
    np.random.shuffle(train_pairs)

    batches = batchify(train_pairs, 8)
    print("batches:", np.array(batches).shape)

    #print(batches[0])
    #print(batches[-1])

    sys.exit(0)
    for i in range(10):
        print(train_pairs[i])
    print(train_pairs[-1])

    #train_betas, _, _ = load_split_betas('2')
    train_betas = np.random.uniform(0, 1, (10000,  327684))
    print("train_betas:", train_betas.shape)

    import data_generator as generator
    init_generator = generator.DataGenerator(
        train_pairs,
        train_betas,
        8,
        tok,
        512,
        15,
        5001,
        pre_load_betas=False,
        shuffle=True,
        training=False)

    x = init_generator[0]
    x = x[0]
    print(len(x))
    print(x[0].shape)
    print(x[1].shape)

    return

if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Total time elapsed: {(time.time() - start):.2f}")


