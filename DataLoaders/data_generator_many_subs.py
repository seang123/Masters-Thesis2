import tensorflow as tf
import numpy as np
import os, sys
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import re
import logging
import time
import tqdm
import warnings
from collections import defaultdict
import concurrent.futures


loggerA = logging.getLogger(__name__ + '.data_generator')

#subject = '2'
nsd_dir = '/home/hpcgies1/rds/hpc-work/NIC/NSD/'
captions_path = "/home/hpcgies1/rds/hpc-work/NIC/Data/captions/"
#betas_path    = f"/fast/seagie/data/subj_{subject}/betas_averaged/"
#guse_path     = f"/fast/seagie/data/subj_{subject}/guse_averaged/"
#vgg16_path    = f"/fast/seagie/data/subj_{subject}/vgg16/"

class DataGenerator(keras.utils.Sequence):
    """ https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly """

    def __init__(self, pairs, betas, batch_size, tokenizer, units, max_len, vocab_size, subject='2', pre_load_betas=False, shuffle=True, training=False):
        print("initialising DataGenerator")
        self.pairs = np.array(pairs) # (n_subs, n_samples, 4)
        self.betas = betas # (n_subs, n_samples, 327684)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.units = units
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.training = training

        print("Generator:")
        print("\tpairs shape:", self.pairs.shape)
        print("\tbetas shape:", self.betas.shape)

        # Setup (call at start)
        self.on_epoch_end()


    def __len__(self):
        """ Nr. of batches per epoch """
        #return len(self.pairs)//self.batch_size
        return len(self.batches)

    def batchify(self, pairs: np.array):
        """ Batchifies the pairs into batches of size self.batch_size
        with each batch holding samples from only 1 sub
        """
        bs = self.batch_size
        batches = []
        for p in range(pairs.shape[0]):
            for i in range(0, len(pairs[p]), bs):
                batches.append(pairs[p][i:i+bs])
        batches = [i for i in batches if len(i) == bs] # drop small batches (for model.fit())

        np.random.shuffle(batches)
        self.batches = batches


    def on_epoch_end(self):
        """ Shuffle data when epoch ends """
        if self.shuffle:
            np.random.shuffle(self.pairs)
            loggerA.info("shuffling dataset")

        self.batchify(self.pairs)
        loggerA.info("Batchifying data")

    def __getitem__(self, index):
        """ Return one batch

        self.batches(list) holds lists of size batch_size (one batch has data from one subject)
        """

        #batch = self.batches[index*self.batch_size:(index+1)*self.batch_size]
        batch = self.batches[index]
        return self.__data_generation(batch)

    def __data_generation(self, batch):
        """ Generates data cointaining batch_size samples

        Takes a batch from the pairs array and returns the appropriate data
        """
        nsd_key, cap, idx, sub = batch[:,0], batch[:,1], batch[:,2], batch[:,3]
        idx = idx.astype(np.int32)
        batch_size = nsd_key.shape[0]

        # Pre-allocate memory
        betas_batch = np.zeros((batch_size, 327684), dtype=np.float32)

        sub_id = {'1':0, '2':1, '5':2, '7':3} # ordering of subjects in the stacked betas np.array (n_subs, 9000, 327684) <- train_betas
        cur_sub_id = sub_id[sub[0]]

        encoder = f"dense_in_{cur_sub_id}"

        # Load data
        for k, i in enumerate(idx):
            betas_batch[k] = self.betas[cur_sub_id, i,:]

        # Tokenize captions
        cap_seqs = self.tokenizer.texts_to_sequences(cap) # int32
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(cap_seqs, maxlen = self.max_len, truncating = 'post', padding = 'post')

        # Create target
        target = np.zeros_like(cap_vector, dtype=cap_vector.dtype)
        target[:,:-1] = cap_vector[:,1:]
        target = to_categorical(target, self.vocab_size)

        # Init LSTM
        init_state = tf.zeros([batch_size, self.units], dtype=np.float32)

        if self.training:
            return ((betas_batch, cap_vector, init_state, init_state), target)
        else:
            return ((betas_batch, cap_vector, init_state, init_state), target, nsd_key)







