import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import yaml
import pandas as pd
import time
import json
import os, sys
import tensorflow as tf
import numpy as np
import tqdm
from Model import lc_NIC
#from Model import tmp_lc_NIC as lc_NIC
from DataLoaders import load_avg_betas2 as loader
from DataLoaders import data_generator as generator
#from DataLoaders import data_generator_ms_sing_enc as generator
#from tabulate import tabulate
import argparse
from itertools import groupby

gpu_to_use = 1
# Allow memory growth on GPU device
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#for i in range(0, len(physical_devices)):
#    tf.config.experimental.set_memory_growth(physical_devices[i], True)
#tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

## ======= Arg parse =========
parser = argparse.ArgumentParser(description='Evaluate NIC model')
parser.add_argument('--dir', type=str, required=False)
parser.add_argument('--e', type=int, required=False)
parser.add_argument('--sub', type=str, required=False)
parser.add_argument('--bs', type=int, required=False) # batch size (for larger models)
args = parser.parse_args()

## ======= Parameters =========
model_name = "no_attn_loss_const_lr2"
epoch = 41
subject = '2'

if args.dir != None: model_name = args.dir
if args.e != None: epoch = args.e
if args.sub != None: subject = args.sub

print(">> Subject: ", subject, " <<")

model_dir = f"./Log/{model_name}/model/model-ep{epoch:03}.h5"
print("Model dir:   ", model_dir)

#print("sleep 5 sec")
#time.sleep(5)

## ======= Config =========

with open(f"./Log/{model_name}/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    print(f"Config file loaded:\n\t {f.name}")

run_name = config['run']
#out_path = os.path.join(config['log'], run_name, 'eval_out')
#out_path = './Eval/one_shot/'
out_path = f'./Log/{run_name}/eval_out'
if not os.path.exists(out_path):
    os.makedirs(out_path)
    print(f"Creating evaluation output dir:\n\t{out_path}")
else:
    print(f"Evaluation output to dir:\n\t{out_path}")

np.random.seed(config['seed'])
tf.random.set_seed(config['seed'])

vocab_size = config['top_k'] + 1
batch_size = 64
if args.bs != None: batch_size = args.bs

## ======= Load data =========

train_keys, val_keys, test_keys = loader.get_nsd_keys(subject)
print("train_keys:", train_keys[:10])
print("train_keys:", train_keys.shape)
print("val_keys:", val_keys.shape)
print("test_keys:", test_keys.shape)

#tokenizer = loader.load_tokenizer()
tokenizer, _ = loader.build_tokenizer(np.arange(1, 73001), config['top_k'])
#tokenizer, _ = loader.build_tokenizer(np.concatenate((train_keys, val_keys)), config['top_k'])

## ===== If training with just 1 subject =======
train_pairs = loader.create_pairs(train_keys, subj=subject, single=True)
val_pairs   = loader.create_pairs(val_keys, subj=subject, single=True)
test_pairs   = loader.create_pairs(test_keys, subj=subject, single=True)
print(f"train_pairs: {len(train_pairs)}")
print(f"val_pairs  : {len(val_pairs)}")
print(f"test_pairs : {len(test_pairs)}")

_, _, test_betas = loader.load_split_betas(subject)
print("test_betas:", test_betas.shape)

## ===== If training on all subjects ======
"""
print("--- loading all subjects data ---")
train_pairs, val_pairs, test_pairs, _, _, test_betas = loader.load_subs([1,2,3,4,5,6,7,8])
test_pairs = np.concatenate(test_pairs, axis=0)
print("test pairs:", test_pairs.shape)
print("test betas:", len(test_betas))
"""


def remove_dup_pairs(pairs):
    """ Remove duplicates from the pairs list, based on NSD key """
    return list({v[0]:v for v in pairs}.values())

## ======= Data Generator =========
data_generator = generator.DataGenerator(
        test_pairs,
        test_betas,
        batch_size,
        tokenizer,
        config['units'],
        config['max_length'],
        vocab_size,
        subject = subject,
        pre_load_betas=False,
        shuffle=False,
        training=False)
print("len generator:", len(data_generator))

## ======= Model =========
model = lc_NIC.NIC(
        loader.get_groups(config['group_size'], True),
        config['units'],
        config['embedding_features'],
        config['embedding_text'],
        config['attn_units'],
        vocab_size,
        config['max_length'],
        config['dropout_input'],
        config['dropout_features'],
        config['dropout_text'],
        config['dropout_attn'],
        config['dropout_lstm'],
        config['dropout_out'],
        config['input_reg'],
        config['attn_reg'],
        config['lstm_reg'],
        config['output_reg']
        )
# Build model
model(data_generator.__getitem__(0)[0], training=False)
print("--= Model built =--")

# Load weights
model.load_weights(model_dir,by_name=True,skip_mismatch=True)
print(f"Model weights loaded")
print(f" - from {model_dir}")


## ======= Evaluate Model =========

def eval_model():
    """ Runs the generators input through the model and returns the output and attention scores """

    all_outputs = []
    all_outputs_raw = []
    all_attention_scores = []
    all_attention_weights = []

    for i in tqdm.tqdm(range(0, len(data_generator)+1)):
        sample = data_generator[i]
        features, _, a0, c0 = sample[0]
        target = sample[1]
        keys = sample[2]

        start_seq = np.repeat([tokenizer.word_index['<start>']], features.shape[0])

        outputs, outputs_raw, attention_scores = model.greedy_predict(features, tf.convert_to_tensor(a0), tf.convert_to_tensor(c0), start_seq, config['max_length'], config['units'], tokenizer)
        all_outputs.append(outputs)
        #all_outputs_raw.append(outputs_raw)
        all_attention_scores.append(attention_scores)
        #all_attention_weights.append(attention_weights)

    # Concat the batches into one matrix
    outputs = np.concatenate((all_outputs), axis=0)
    #outputs_raw = np.concatenate((all_outputs_raw), axis=0)
    attention_scores = np.swapaxes(np.concatenate((all_attention_scores), axis=1), 0, 1)
    #attention_weights = np.swapaxes(np.concatenate((all_attention_weights), axis=1), 0, 1)

    print("outputs:", outputs.shape)
    #print("outputs_raw:", outputs_raw.shape)
    print("attention scores:", attention_scores.shape)

    add_name = f""

    with open(f"{out_path}/output_captions_{epoch}{add_name}.npy", "wb") as f:
        np.save(f, outputs)
    #with open(f"{out_path}/output_captions_raw_{epoch}{add_name}.npy", "wb") as f:
    #    np.save(f, outputs_raw)
    with open(f"{out_path}/attention_scores_{epoch}{add_name}.npy", "wb") as f:
        np.save(f, attention_scores)
    #with open(f"{out_path}/attention_weights_{epoch}{add_name}.npy", "wb") as f:
    #    np.save(f, attention_weights)
    with open(f"{out_path}/tokenizer.json", "w") as f:
        f.write(tokenizer.to_json())


    return outputs, attention_scores

def eval_fc_model():
    """ Evaluate the FC model """
    all_outputs = []
    all_outputs_raw = []
    all_attention_scores = []
    all_attention_weights = []

    for i in tqdm.tqdm(range(0, len(data_generator)+1)):
        sample = data_generator[i]
        features, _, a0, c0 = sample[0]
        target = sample[1]
        keys = sample[2]

        start_seq = np.repeat([tokenizer.word_index['<start>']], features.shape[0])

        outputs = model.greedy_predict(features, tf.convert_to_tensor(a0), tf.convert_to_tensor(c0), start_seq, config['max_length'], config['units'], tokenizer)
        all_outputs.append(outputs)

    all_outputs = np.swapaxes(np.concatenate((all_outputs), axis=1), 0, 1)
    print("outputs:", all_outputs.shape)

    add_name = ""
    with open(f"{out_path}/output_captions_{epoch}{add_name}.npy", "wb") as f:
        np.save(f, all_outputs)
    with open(f"{out_path}/tokenizer.json", "w") as f:
        f.write(tokenizer.to_json())

if __name__ == '__main__':
    #eval_fc_model()
    eval_model()
