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
import argparse

## ======= Arg parse =========
parser = argparse.ArgumentParser(description='NIC model')
parser.add_argument('--train', type=int, default=1, required=False) # False for inference
args = parser.parse_args()


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
training = True
if args.train == 0:
    training = False
print(f"-- Training = {training} --")

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
def init_dataset(subs: list = [1,2,3,4,5,6,7,8]):
    train_pairs, val_pairs, test_pairs, train_betas, val_betas, test_betas = loader.load_subs(subs) # (subs 4 and 8 val set is == test set)

    # Synthetic data (for testing)
    #train_pairs, train_betas = testing()

    n_subjects = len(train_pairs)

    train_generators = {}
    val_generators   = {}
    test_generators  = {}

    for i in range(n_subjects):
        train_dataset = Dataset(train_pairs[i], train_betas[str(i+1)], tokenizer, config['units'], config['max_length'], vocab_size, device)
        train_generators[str(i+1)] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    if n_subjects > 2:
        val_subjects = ['1', '2', '3', '5', '6', '7'] # sub 4 and 8 have no val set
        for i in val_subjects:
            val_dataset = Dataset(val_pairs[int(i)-1], val_betas[i], tokenizer, config['units'], config['max_length'], vocab_size, device)
            val_generators[i] = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    for i in range(n_subjects):
        test_dataset = Dataset(test_pairs[i], test_betas[str(i+1)], tokenizer, config['units'], config['max_length'], vocab_size, device)
        test_generators[str(i+1)] = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_generators, val_generators, test_generators, n_subjects


if training:
    train_generators, val_generators, _, n_subjects = init_dataset([1,2,3,4,5,6,7,8])
else:
    train_generators, _, test_generators, n_subjects = init_dataset([1,2,3,4,5,6,7,8])
print(">> N subjects:", n_subjects, "<<")


#
# ------------- Model ------------
#
model = NIC(groups, 32, 512, 512, config['max_length'], vocab_size, n_subjects=n_subjects).to(device)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), weight_decay=3.0e-5)
#for k, v in enumerate(train_generators['1']):
    #print(summary(model, input_data={'x':v[:-1], 'subject':1}, batch_dim=64, verbose=0))
    #break

with torch.no_grad():
    for sub in range(8):
        generator = train_generators[str(sub+1)]
        for batch_nr, data in enumerate(generator):
            features, captions, hidden, carry, target = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)

            # Model pass
            output, _, loss_dict = model.train_step((features, captions, hidden, carry), target, subject=sub) # (bs, max_len, vocab_size)
            break
        print(f"Model initalised on sub: {sub+1}")




def L2_loss(parameters, alpha):
    loss = 0
    for param in parameters:
        if param.dim() >= 2 and param.shape[0] == 32:
            loss += torch.sum(torch.pow(param, 2))
    return alpha * loss


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

            for v_sub in val_subjects.keys():
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


def load_model(model, path):
    """ Load model checkpoint """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    try:
        loss  = checkpoint['loss']
    except KeyError:
        loss = np.nan
    return model, epoch, loss

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
    model.eval()

    start_time = time.time()
    outputs = []
    attention_maps = []

    for sub in range(n_subjects):

        sub_outputs = []
        sub_attention_maps = []
        generator = generators[str(sub+1)]
        print(f"Inference // subject: {sub+1}")

        for batch_nr, data in enumerate(generator):

            features, captions, hidden, carry, target = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4]

            # Start word
            start_seq = np.repeat([tokenizer.word_index['<start>']], features.shape[0])
            start_seq = torch.from_numpy(start_seq).to(device)

            # Generate caption
            output, attn_map = model.predict(features, start_seq, hidden, carry, subject=sub) # out:

            print("output:", output.shape)
            print("attn maps:", attn_map)
            raise

            sub_outputs.append(output)
            sub_attention_maps.append(attn_map)
        outputs.append(sub_outputs)
        attention_maps.append(sub_attention_maps)

    # Outputs to ndarray
    outputs = np.concatenate(outputs, axis=0)
    atttention_maps = np.concatenate(attention_maps, axis=1)
    print("outputs:", outputs.shape)
    print("attention maps:", attention_maps.shape)

    # Save outputs
    np.save(open(f"{out_path}/output_captions_{epoch}{add_name}.npy", "wb"), outputs)
    np.save(open(f"{out_path}/attention_scores_{epoch}{add_name}.npy", "wb"), attention_maps)
    with open(f"{out_path}/tokenizer.json", "w") as f:
        f.write(tokenizer.to_json())

    print(f"Inference complete! ({(time.time() - start_time):.1f}s)")


def inference(model, path, generators: dict):
    """ Loads a model and runs inference on the given generators """
    # Load model
    model, _, _ = load_model(model, path)
    # Run inference
    generate_predictions(model, generators)
    return


if __name__ == '__main__':

    if training:
        train(model, optimizer, criterion)
        print("Done.")
    else:
        print("Running inference")
        inference(model, "/home/hpcgies1/rds/hpc-work/NIC/Log/multi_subject_torch3/model/model_ep025.pt", test_generators)






