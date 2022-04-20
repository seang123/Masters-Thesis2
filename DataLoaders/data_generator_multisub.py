import tensorflow as tf
import numpy as np
import os, sys
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import re
import logging
import time
import warnings
import concurrent.futures

"""
Same as data generator guse but for multiple subjects

Split batches into equal parts for each subject (batch size must evenly fit the subjects)
"""


loggerA = logging.getLogger(__name__ + '.data_generator_ms')

nsd_dir = '/home/seagie/NSD2/'
captions_path = "/fast/seagie/data/subj_2/captions/"
#betas_path    = "/fast/seagie/data/subj_2/betas_averaged/"
guse_path     = "/fast/seagie/data/subj_2/guse_averaged/"
vgg16_path    = "/fast/seagie/data/subj_2/vgg16/"

class DataGenerator(keras.utils.Sequence):
    """ https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly """

    def __init__(self, pairs, betas,  batch_size, tokenizer, units, max_len, vocab_size, pre_load_betas=False, shuffle=True, training=False):
        print("initialising DataGenerator -- multisubject")
        self.batch_size = batch_size
        self.betas = betas
        self.tokenizer = tokenizer
        self.units = units
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.training = training
        self.pre_load_betas = pre_load_betas

        pairs = np.array(pairs) # (2, n_samples=45000) idx 0 is subject 1, idx 1 is subj2
        self.n_pairs = pairs.shape[0]
        self.pairsA = pairs[0]
        self.pairsB = pairs[1]
        print("pairsA:", self.pairsA.shape)
        print("pairsB:", self.pairsB.shape)
        assert self.pairsA.shape == self.pairsB.shape, "Subjects need to have equal data split"

        #assert batch_size % 2 == 0, "Batch size needs to evenly divide two subjects"
        #self.batch_size = self.batch_size // 2
        self.on_epoch_end()

    def load_all_betas(self, keys):
        # Not corrently implemented when using more than one subject
        betas = np.zeros((keys.shape[0], 327684), dtype=np.float32)
        for i, key in enumerate(keys):
            with open(f"{betas_path}/subj02_KID{key}.npy", "rb") as f:
                betas[i,:] = np.load(f)

        loggerA.info("betas pre-loaded into memory")
        return betas

    def __len__(self):
        """ Nr. of batches per epoch """
        return self.pairsA.shape[0]//self.batch_size

    def on_epoch_end(self):
        """ Shuffle data when epoch ends """
        if self.shuffle:
            np.random.shuffle(self.pairsA)
            np.random.shuffle(self.pairsB)
            loggerA.info("shuffling dataset")

    def __getitem__(self, index):
        """ Return one batch """

        batchA = self.pairsA[index*self.batch_size:(index+1)*self.batch_size]
        batchB = self.pairsB[index*self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(batchA, batchB)

    def __data_generation(self, batchA, batchB):
        """ Generates data cointaining batch_size samples

        Takes a batch from the pairs array and returns the appropriate data
        """

        #nsd_keyA, capA, _, countA, subj_idA = batchA[:,0], batchA[:,1], batchA[:,2], batchA[:,3], batchA[:,4]
        #nsd_keyB, capB, _, countB, subj_idB = batchB[:,0], batchB[:,1], batchB[:,2], batchB[:,3], batchB[:,4]
        nsd_keyA, capA, idxA, subA = batchA[:,0], batchA[:,1], batchA[:,2], batchA[:,3]
        nsd_keyB, capB, idxB, subB = batchB[:,0], batchB[:,1], batchB[:,2], batchB[:,3]
        idxA = idxA.astype(np.int32)
        idxB = idxB.astype(np.int32)

        caps = np.concatenate((capA, capB), axis=0)
        nsd_keys = np.concatenate((nsd_keyA, nsd_keyB), axis=0)

        betas_batch = np.zeros((nsd_keys.shape[0], 327684), dtype=np.float32)

        ## Load betas
        # First half is subject 1, second half is subject 2
        betas_batch[0:nsd_keys.shape[0]//2, :] = self.betas[0,idxA, :]
        betas_batch[nsd_keys.shape[0]//2:, :] = self.betas[1,idxB, :]


        # Tokenize captions
        cap_seqs = self.tokenizer.texts_to_sequences(caps) # int32
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(cap_seqs, maxlen = self.max_len, truncating = 'post', padding = 'post')

        # Create target
        target = np.zeros_like(cap_vector, dtype=cap_vector.dtype)
        target[:,:-1] = cap_vector[:,1:]
        target = to_categorical(target, self.vocab_size)

        # Init LSTM
        init_state = tf.zeros([nsd_keys.shape[0], self.units], dtype=np.float32)

        if self.training:
            return ((betas_batch, cap_vector, init_state, init_state), target)
        else:
            return ((betas_batch, cap_vector, init_state, init_state), target, nsd_keys)






