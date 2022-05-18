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
import pandas as pd
from DataLoaders import load_avg_betas2 as loader
from Model.torch_model import NIC
from DataLoaders.torch_generator import Dataset
from torchinfo import summary


print("cuda is available:", torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)


thesis_dir = "/home/hpcgies1/Masters-Thesis/AttemptFour/"

## Load the configuration file
with open(f"{thesis_dir}/torch_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    print(f"Config file loaded.", flush=True)

np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
torch.cuda.manual_seed(config['seed'])
#torch.backends.cudnn.benchmarks = True # Optimize graphs - requires that input/output don't change in size

run_name = config['run']
run_path = os.path.join(config['dataset']['log'], run_name)
print("Run path:", run_path)

if not os.path.exists(run_path):
    os.makedirs(run_path)
    print(f"Creating training Log folder: {run_path}", flush=True)
else:
    print(f"Training Log will be saved to: {run_path}", flush=True)

if not os.path.exists(os.path.join(run_path, 'model')):
    os.makedirs(os.path.join(run_path, 'model'))


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
batch_size = config['batch_size']
print("--==Parameters==--")
print("epochs:", epochs)
print("batch size:", batch_size)
print("vocab size:", vocab_size)
print("max length:", max_len)

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

#train_pairs, val_pairs, test_pairs, train_betas, val_betas, _ = loader.load_subs([1,2,3,4,5,6,7,8]) # (subs 4 and 8 val set is == test set)
train_pairs, train_betas = testing()

n_subjects = len(train_pairs)
print(">> N subjects:", n_subjects, "<<")

## Create the data generators for each subject
train_generators = {}
for i in range(n_subjects):
    train_dataset = Dataset(train_pairs[i], train_betas[str(i+1)], tokenizer, config['units'], config['max_length'], vocab_size, device)
    train_generators[str(i+1)] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

"""
val_generators = {}
val_subjects = ['2', '7']
for i in val_subjects:
    val_dataset = Dataset(val_pairs[int(i)-1], train_betas[i], tokenizer, config['units'], config['max_length'], vocab_size, device)
    val_generators[i] = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
"""


#
# ------------- Model ------------
#
model = NIC(groups, 32, 512, 512, config['max_length'], vocab_size, n_subjects=n_subjects).to(device)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), weight_decay=3.0e-5)
#for k, v in enumerate(train_generators['1']):
    #print(summary(model, input_data={'x':v[:-1], 'subject':1}, batch_dim=64, verbose=0))
    #break




def cross_entropy(pred, target):
    """ Compute cross entropy between two distributions """
    return torch.mean(-torch.sum(target * torch.log(pred), dim=1))# (bs, 5001) -> (64) -> (1)

def accuracy(pred, target):
    target_arg_max = torch.argmax(target, dim=1)
    pred_arg_max   = torch.argmax(pred, dim=1)
    return torch.sum(pred_arg_max == target_arg_max) # should be (bs,)

def train(model, optimizer, criterion):

    #losses = defaultdict(list)
    for epoch in range(1,epochs+1):
        print(f" ---== Epoch :: {epoch} ==---")

        losses = []
        epoch_time = time.time()
        for sub in range(n_subjects):

            generator = train_generators[str(sub+1)]
            sub_loss = defaultdict(list)
            sub_start_time = time.time()

            model.train()
            # Loop through the dataset
            for batch_nr, data in enumerate(generator):
                batch_time = time.time()
                # Access data
                features, captions, hidden, carry, target = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)

                # DataLoader time
                prepare_time = batch_time-time.time()

                # Zero gradients
                optimizer.zero_grad()

                # Model pass
                output, _, loss_dict = model.train_step((features, captions, hidden, carry), target, subject=sub) # (bs, max_len, vocab_size)

                # Model forward pass time
                model_time = batch_time - prepare_time - time.time()

                # Backprop
                loss = loss_dict['loss']
                loss.backward()
                optimizer.step()

                # Store loss
                loss_ = loss_dict['loss'].detach().item()
                acc_  = loss_dict['accuracy'].detach().item()
                sub_loss['sub_epoch_loss'].append(loss_)
                sub_loss['sub_epoch_acc'].append(acc_)
                losses.append( {'epoch':epoch, 'sub': sub+1, 'batch': batch_nr, 'loss':loss_, 'accuracy':acc_} )

                process_time = batch_time - time.time() - prepare_time
                # Print information
                if batch_nr % 25 == 0:
                    print(f"epoch: {epoch:02}/{epochs} | sub: {sub} | {batch_nr:03} | loss: {np.mean(sub_loss['sub_epoch_loss']):.3f} | acc: {np.mean(sub_loss['sub_epoch_acc']):.3f} | comp. eff.: {(model_time/(model_time+prepare_time)):.3f}")

            # Store loss
            print(f"--- sub: {sub} // time: {(time.time()-sub_start_time):.2f}s // loss: {np.mean(sub_loss['sub_epoch_loss']):.3f} ---")

        # ----------
        # Validation
        # ----------
        model.eval()
        with torch.no_grad():

            for v_sub in val_subjects:
                val_loss = defaultdict(list)
                val_generator = val_generators[v_sub]

                for batch_nr, data in enumerate(val_generator):

                    # Data prep
                    features, captions, hidden, carry, target = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)

                    # Model pass
                    output, _, loss_dict = model.train_step((features, captions, hidden, carry), target, subject=int(v_sub)-1) # (bs, max_len, vocab_size)

                    # Losses
                    loss_ = loss_dict['loss'].detach().item()
                    acc_  = loss_dict['accuracy'].detach().item()
                    val_loss['loss'].append( loss_ )
                    val_loss['accuracy'].append( acc_ )
                    losses.append( {'epoch':epoch, 'sub': v_sub, 'batch': batch_nr, 'val_loss': loss_, 'val_accuracy':acc_} )

                print(f"validation: {epoch:02}/{epochs} | sub: {v_sub} | loss: {np.mean(val_loss['loss']):.3f} | accuracy: {np.mean(val_loss['accuracy']):.3f}")

        ## ----- Post epoch -----

        # Store Loss
        pd.DataFrame(losses).to_csv(f"{run_path}/loss_history.csv", mode='a', header=not os.path.exists(f"{run_path}/loss_history.csv"))
        # Save model
        torch.save({'epoch':epoch, 'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}, f'{run_path}/model/model_ep{epoch:03}.pt')

        print(f"epoch: {epoch:02}/{epochs} complete! ({(time.time()-epoch_time):.1f}s)\n")


    # ---- Post training ----



def generate_predictions(model, generators: dict):
    """ Generate predictions

    Parameters:
    -----------
        model - pytorch-model
        generators - dict
            dictionary of pytorch generators

    Return:
    -------
        output : np.array
            (n_subjects, 515, max_len, 1) integer encoded output captions
        attention maps : np.array
            (n_subject, 515, max_len, 360, 1) attention maps
    """


    for sub in range(n_subjects):
        generator = generators[str(sub+1)]
        print("Inference // subject: {sub+1}")
        for batch_nr, data in enumerate(generator):
            features, captions, hidden, carry, target = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4]

            # Generate caption
            output, attention_maps = model.predict((features, start_seq, hidden, carry), subject=sub) # out:



if __name__ == '__main__':
    train(model, optimizer, criterion)
    print("Done.")






