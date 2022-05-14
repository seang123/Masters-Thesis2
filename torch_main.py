import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from datetime import datetime
from collections import defaultdict
import os, sys, time
import logging
from DataLoaders import load_avg_betas2 as loader
from Model.torch_model import NIC
from DataLoaders.torch_generator import Dataset


print("cuda is available:", torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)

thesis_dir = "/home/hpcgies1/Masters-Thesis/AttemptFour/"

## Load the configuration file
with open(f"{thesis_dir}/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    print(f"Config file loaded.", flush=True)

run_name = config['run']
run_path = os.path.join(config['log'], run_name)

if not os.path.exists(run_path):
    os.makedirs(run_path)
    print(f"Creating training Log folder: {run_path}", flush=True)
else:
    print(f"Training Log will be saved to: {run_path}", flush=True)

with open(f"{run_path}/config.yaml", "w+") as f:
    yaml.dump(config, f)

np.random.seed(config['seed'])

#
## ---------- Parameters ----------
#
vocab_size = config['top_k'] + 1
groups = loader.get_groups(config['group_size'], separate_hemi=True)
max_len = config['max_length']
epochs = config['epochs']
batch_size = 64

#
## ---------- Load Data ----------
tokenizer, _ = loader.build_tokenizer(np.arange(1, 73001), top_k=5000)

def testing():
    train_pairs = []
    val_pairs = []
    for sub in range(2):
        t_ = []
        for i in range(900):
            t_.append( [sub, f"<start> hello this is a test <end>", sub, i**2] )
        train_pairs.append(t_)
    for sub in range(2):
        t_ = []
        for i in range(485):
            t_.append( [sub, f"bye this is a test", sub, i**2] )
        val_pairs.append(t_)

    train_pairs = np.array(train_pairs)
    print("test len trian pairs:", train_pairs.shape)
    train_betas = {}
    train_betas['1'] = np.random.uniform(0, 1, (900, 327684))
    train_betas['2'] = np.random.uniform(0, 1, (485, 327684))
    return train_pairs, train_betas

print("------ Data ------")
train_pairs, val_pairs, test_pairs, train_betas, val_betas, _ = loader.load_subs([1,2])#,3,4,5,6,7,8]) # (subs 4 and 8 val set is == test set)
#train_pairs = np.concatenate(train_pairs, axis=0)
#test_pairs = np.concatenate(test_pairs, axis=0)
#val_pairs = np.concatenate((val_pairs[0], val_pairs[2], val_pairs[4], val_pairs[6]), axis=0) # 1 3 5 7
#val_pairs = np.concatenate(val_pairs, axis=0)

#train_pairs, train_betas = testing()

n_subjects = len(train_pairs)

## Create the data generators for each subject
train_generators = {}
for i in range(n_subjects):
    train_dataset = Dataset(train_pairs[i], train_betas[str(i+1)], tokenizer, config['units'], config['max_length'], vocab_size, device)
    train_generators[str(i+1)] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

#
# ------------- Model ------------
#
model = NIC(groups, 32, 512, 512, config['max_length'], vocab_size)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1.0e-10)


def cross_entropy(pred, target):
    """ Compute cross entropy between two distributions """
    return -torch.sum(target * torch.log(pred))

def accuracy(pred, target):

    target_arg_max = torch.argmax(target, dim=1)
    pred_arg_max   = torch.argmax(pred, dim=1)
    return torch.sum(pred_arg_max == target_arg_max)

def train(model, optimizer, criterion):

    losses = defaultdict(list)
    for epoch in range(1,epochs+1):
        for sub in range(n_subjects):
            print(f"epoch: {epoch} // subj: {sub}")
            generator = train_generators[str(sub+1)]
            sub_loss = defaultdict(list)
            sub_start_time = time.time()
            for batch_nr, data in enumerate(generator):
                features, captions, hidden, carry, target = data
                optimizer.zero_grad()

                # Model
                output = model((features, captions, hidden, carry), subject=sub) # (bs, max_len, vocab_size)

                # Compute loss
                loss = 0
                acc = 0
                for i in range(max_len):
                    loss += cross_entropy(output[:,i,:], target[:,i,:])
                    acc += accuracy(output[:,i,:], target[:,i,:]).float()

                loss /= max_len
                acc /= max_len

                # Backprop
                loss.backward()
                optimizer.step()

                loss_detach = loss.detach()
                acc_detach = acc.detach()
                sub_loss['train_loss'].append(loss_detach.item())

                if batch_nr % 100 == 0:
                    print(f"epoch: {epoch:02}/{epochs} | {batch_nr:03} | loss: {loss_detach.cpu().numpy():.3f} | acc: {acc_detach:.3f}")

            print(f"epoch: {epoch:02}/{epochs} | sub: {sub} | time: {(time.time()-sub_start_time):.2f}s | loss: {np.mean(sub_loss['train_loss']):.3f}")


if __name__ == '__main__':
    train(model, optimizer, criterion)
    print("Done.")






