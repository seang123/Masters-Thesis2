
import numpy as np
import os, sys
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import torch

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
        betas = torch.from_numpy(self.betas[idx,:].astype(np.float32)) # [327684]

        # Tokenize caption
        cap_seqs = self.tokenizer.texts_to_sequences([cap])
        cap_vector = pad_sequences(cap_seqs, maxlen=self.maxlen, truncating='post', padding='post')[0] # [15]

        # Create target
        target = np.zeros_like(cap_vector, dtype=cap_vector.dtype)
        target[:-1] = cap_vector[1:]
        target = to_categorical(target, self.vocab_size) # [15]

        # init state
        init_state = torch.zeros(self.units, dtype=torch.float32) # [512]

        target = torch.from_numpy(target)
        cap_vector = torch.from_numpy(cap_vector)

        return betas, cap_vector, init_state, init_state, target
