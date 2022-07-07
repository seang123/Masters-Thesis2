import numpy as np
import os
import sys
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import torch


class Dataset_images(torch.utils.data.Dataset):
    """ Dataset for images """

    def __init__(self, pairs, images, tokenizer, units, maxlen, vocab_size, batch_size, device):
        self.pairs=np.array(pairs)
        self.images=images
        self.tokenizer = tokenizer
        self.units = units
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.device = device
        self.batch_size = batch_size
        print(f"Dataset size: {self.__len__()} (batches)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        img_id, idx, cap = self.pairs[index]
        idx = idx.astype(np.int32)

        # Get beta
        imgs  = torch.from_numpy(self.images[idx].astype(np.float32))

        # Tokenize caption
        cap_seqs = self.tokenizer.texts_to_sequences([cap])
        cap_vector = pad_sequences(cap_seqs, maxlen=self.maxlen, truncating='post', padding='post')[0]  # [15]

        # Create target
        target = np.zeros_like(cap_vector, dtype=cap_vector.dtype)
        target[:-1] = cap_vector[1:]
        target = to_categorical(target, self.vocab_size)  # [15]

        # init state
        init_state = torch.zeros(self.units, dtype=torch.float32)  # [512]

        target = torch.from_numpy(target)
        cap_vector = torch.from_numpy(cap_vector)

        return imgs, cap_vector, init_state, init_state, target





class Dataset_batch(torch.utils.data.Dataset):
    """ Batched model with alternating batches from different subjects """

    def __init__(self, pairs, betas, tokenizer, units, maxlen, vocab_size, batch_size, device):
        self.pairs = pairs  # flattened pairs [n_subs * n_trials * 5]
        self.betas = betas  # Dictionary of np.arrays {subject: betas}
        self.tokenizer = tokenizer
        self.units = units
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.device = device
        self.batch_size = batch_size
        print(f"Dataset size: {self.__len__()} (batches)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # Get sample information
        pair = np.array(self.pairs[index])
        nsd_key, cap, idx, sub = pair[:,0], pair[:,1], pair[:,2], pair[:,3]
        idx = idx.astype(np.int32)
        sub = sub[0]

        # Get beta
        betas = torch.from_numpy(self.betas[sub][idx, :].astype(np.float32))  # [327684]

        # Sub to Tensor (str -> IntTensor)
        sub = torch.IntTensor([int(sub)])

        # Tokenize caption
        cap_seqs = self.tokenizer.texts_to_sequences(cap)
        cap_vector = pad_sequences(cap_seqs, maxlen=self.maxlen, truncating='post', padding='post')  # [15]

        # Create target
        target = np.zeros_like(cap_vector, dtype=cap_vector.dtype)
        target[:, :-1] = cap_vector[:, 1:]
        target = to_categorical(target, self.vocab_size)  # [15]

        # init state
        init_state = torch.zeros([betas.shape[0], self.units], dtype=torch.float32)  # [512]

        target = torch.from_numpy(target)
        cap_vector = torch.from_numpy(cap_vector)

        return betas, cap_vector, init_state, init_state, sub, target

class Dataset_mix(torch.utils.data.Dataset):
    """ Takes all data and returns random mixed batches """

    def __init__(self, pairs, betas, tokenizer, units, maxlen, vocab_size, device):
        self.pairs = pairs  # flattened pairs [n_subs * n_trials * 5]
        self.betas = betas  # Dictionary of np.arrays {subject: betas}
        self.tokenizer = tokenizer
        self.units = units
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.device = device

        print("Dataset_mix")
        print("pair:", len(self.pairs))
        print("betas:", betas.keys())

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # Get sample information
        nsd_key, cap, idx, sub = self.pairs[index]
        idx = idx.astype(np.int32)
        # Get beta
        betas = torch.from_numpy(self.betas[sub][idx, :].astype(np.float32))  # [327684]

        # Sub to Tensor (str -> IntTensor)
        sub = torch.IntTensor([int(sub)])

        # Tokenize caption
        cap_seqs = self.tokenizer.texts_to_sequences([cap])
        cap_vector = pad_sequences(cap_seqs, maxlen=self.maxlen, truncating='post', padding='post')[0]  # [15]

        # Create target
        target = np.zeros_like(cap_vector, dtype=cap_vector.dtype)
        target[:-1] = cap_vector[1:]
        target = to_categorical(target, self.vocab_size)  # [15]

        # init state
        init_state = torch.zeros(self.units, dtype=torch.float32)  # [512]

        target = torch.from_numpy(target)
        cap_vector = torch.from_numpy(cap_vector)

        return betas, cap_vector, init_state, init_state, sub, target


class Dataset(torch.utils.data.Dataset):

    def __init__(self, pairs, betas, tokenizer, units, maxlen, vocab_size, device):
        self.pairs = pairs
        self.betas = betas
        self.tokenizer = tokenizer
        self.units = units
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.device = device

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        """ Returns a single sample

        Gets called repeatedly by torch generator to generate a single batch
        """

        # Get sample information
        nsd_key, cap, idx, sub = self.pairs[index]
        idx = idx.astype(np.int32)
        # Get beta
        betas = torch.from_numpy(self.betas[idx, :].astype(np.float32))  # [327684]

        # Tokenize caption
        cap_seqs = self.tokenizer.texts_to_sequences([cap])
        cap_vector = pad_sequences(cap_seqs, maxlen=self.maxlen, truncating='post', padding='post')[0]  # [15]

        # Create target
        target = np.zeros_like(cap_vector, dtype=cap_vector.dtype)
        target[:-1] = cap_vector[1:]
        target = to_categorical(target, self.vocab_size)  # [15]

        # init state
        init_state = torch.zeros(self.units, dtype=torch.float32)  # [512]

        target = torch.from_numpy(target)
        cap_vector = torch.from_numpy(cap_vector)

        return betas, cap_vector, init_state, init_state, target
