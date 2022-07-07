import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from datetime import datetime
from collections import defaultdict
import os
import sys
import time
import logging
import pandas as pd
from DataLoaders import load_avg_betas2 as loader
from Model.torch_model import NIC
from DataLoaders.torch_generator import Dataset, Dataset_mix, Dataset_batch, Dataset_images
from torchinfo import summary
import argparse
from rich.console import Console
console = Console()

"""
    Trains on images using VGG16 last conv layer output (196, 512)

"""

## ======= Arg parse =========
parser = argparse.ArgumentParser(description='NIC model')
#parser.add_argument('--train', type=int, default=1, required=False)  # False for inference
parser.add_argument('--eval', action='store_true')
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
run_path = os.path.join(config['log'], run_name)
print("Run path:", run_path)

out_path = f"{run_path}/eval_out/"
print("Out path:", out_path)


# Create log folder
if not os.path.exists(run_path):
    os.makedirs(run_path)
    print(f"Creating training Log folder: {run_path}", flush=True)
else:
    print(f"Training Log will be saved to: {run_path}", flush=True)


# Create checkpoint folder
if not os.path.exists(os.path.join(run_path, 'model')):
    os.makedirs(os.path.join(run_path, 'model'))


# Create eval_out dir
if not os.path.exists(out_path):
    os.makedirs(out_path)

with open(f"{run_path}/config.yaml", "w+") as f:
    yaml.dump(config, f)

np.random.seed(config['seed'])

#
## ---------- Parameters ----------
#
training = True
if args.eval:
    training = False
#print(f"-- Training = {training} --")
console.print( str(("Training = [bold red]False[/bold red]" if training==False else "Training = [bold green]True[/bold green]")) , highlight=False)

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




#
#
#  Load and process data
#
#



""" Create a dataset for the images """

# Encoded images
train_images = np.load("/home/hpcgies1/rds/hpc-work/NIC/Data/Images/mscoco_2017/train_vgg16.npy")
val_images   = np.load("/home/hpcgies1/rds/hpc-work/NIC/Data/Images/mscoco_2017/val_vgg16.npy")
print("train_images:", train_images.shape)
print("val_images:", val_images.shape)

# Captions
train_captions = pd.read_csv("/home/hpcgies1/rds/hpc-work/NIC/Data/Images/mscoco_2017/train_id_to_captions.csv")
val_captions = pd.read_csv("/home/hpcgies1/rds/hpc-work/NIC/Data/Images/mscoco_2017/val_id_to_captions.csv")

# Image_id -> index
train_id_to_idx = pd.read_csv("/home/hpcgies1/rds/hpc-work/NIC/Data/Images/mscoco_2017/train_img_id_to_idx.csv")
val_id_to_idx = pd.read_csv("/home/hpcgies1/rds/hpc-work/NIC/Data/Images/mscoco_2017/val_img_id_to_idx.csv")

# Sub2 cocoId's
sub2_ids = set(list(pd.read_csv("/home/hpcgies1/rds/hpc-work/NIC/Data/Images/mscoco_2017/nsd_key_to_cocoID_sub2.csv")['cocoId'].values))

## Create pairs
# [cocoId, idx, caption]
def create_pairs(captions_df, id_to_idx):
    pairs = []
    cocoID = list(captions_df['cocoId'].values)

    id_to_cap = defaultdict(list)
    for i in range(captions_df.shape[0]):
        temp = captions_df.iloc[i].values
        id_to_cap[temp[0]] = temp[1:]

    for t in cocoID:
        idx_ = id_to_idx['idx'].loc[id_to_idx['cocoId'] == t].values[0]
        caps = id_to_cap[t]
        for c in caps:
            pairs.append( [t, idx_, c] )
    return pairs

train_pairs = create_pairs(train_captions, train_id_to_idx)
val_pairs = create_pairs(val_captions, val_id_to_idx)
print("train_pairs:", len(train_pairs))
print("val_pairs:", len(val_pairs))

## Remove Sub2 samples
train_pairs = [i for i in train_pairs if i[0] not in sub2_ids]
val_pairs = [i for i in val_pairs if i[0] not in sub2_ids]
print("train_pairs (w/o sub2):", len(train_pairs))
print("val_pairs (w/o sub2):", len(val_pairs))

for i in range(11):
    print(train_pairs[i])
print()
for i in range(11):
    print(val_pairs[i])

train_dataset = Dataset_images(train_pairs, train_images, tokenizer, config['units'], config['max_length'], vocab_size, batch_size, device)
train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

val_dataset = Dataset_images(val_pairs, val_images, tokenizer, config['units'], config['max_length'], vocab_size, batch_size, device)
val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)






# ------------- Model ------------
#
model = NIC(
    groups,
    32,  # feat
    512, # text
    512, # units
    config['max_length'],
    vocab_size,
    subjects=[1, 2, 3, 4, 5, 6, 7, 8]).to(device)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), weight_decay=3.0e-4)  # 0.0003




def L2_loss(parameters, alpha):
    loss = 0
    for param in parameters:
        if param.dim() >= 2 and param.shape[0] == 32:
            loss += torch.sum(torch.pow(param, 2))
    return alpha * loss


def cross_entropy(pred, target):
    """ Compute cross entropy between two distributions """
    return torch.mean(-torch.sum(target * torch.log(pred), dim=1))  # (bs, 5001) -> (64) -> (1)


def accuracy(pred, target):
    target_arg_max = torch.argmax(target, dim=1)
    pred_arg_max   = torch.argmax(pred, dim=1)
    return torch.sum(pred_arg_max == target_arg_max)  # should be (bs,)


def train(model, optimizer, criterion):
    """ Trains on alternating batches """

    losses = []
    for epoch in range(1, epochs + 1):
        print(f" ---== Epoch :: {epoch} ==---")
        epoch_loss = defaultdict(list)

        model.train()
        for batch_nr, data in enumerate(train_generator):
            batch_time = time.time()
            features, captions, hidden, carry, target = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
            #subject = str(subject.item())
            subject = '0'

            # DataLoader time
            prepare_time = batch_time - time.time()

            # Zero gradients
            optimizer.zero_grad()

            # Model pass
            output, _, loss_dict = model.train_step((features, captions, hidden, carry), target, subject=subject)  # (bs, max_len, vocab_size)

            # Model forward pass time
            model_time = batch_time - prepare_time - time.time()

            # L2 loss - encoder
            #l2_loss_enc = L2_loss(model.parameters(), 0.01)

            # Backprop
            loss = loss_dict['loss']
            total_loss = loss #+ l2_loss_enc  # Add encoder l2
            total_loss.backward()
            optimizer.step()

            # Store loss data
            loss_ = loss.detach().item()
            acc_  = loss_dict['accuracy'].detach().item()
            l2_loss_enc_ = 0 #l2_loss_enc.detach().item()

            epoch_loss['loss'].append(loss_)
            epoch_loss['accuracy'].append(acc_)
            epoch_loss['L2'].append(l2_loss_enc_)
            losses.append({'epoch': epoch, 'batch': batch_nr, 'subject': subject, 'loss': loss_, 'enc-L2': l2_loss_enc_, 'accuracy': acc_})

            process_time = batch_time - time.time() - prepare_time
            # Print information
            if batch_nr % 25 == 0:
                print(f"epoch: {epoch:02}/{epochs} | {batch_nr:03} | loss: {np.mean(epoch_loss['loss']):.3f} | enc-L2: {np.mean(epoch_loss['L2']):.3f} | acc: {np.mean(epoch_loss['accuracy']):.3f} | comp. eff.: {(process_time/(process_time+prepare_time)):.3f} | {(time.time()-batch_time):.2f} sec")
        print(f"Epoch: {epoch} complete!")

        # Evaluation
        # ----------
        model.eval()
        with torch.no_grad():
            for batch_nr, data in enumerate(val_generator):
                features, captions, hidden, carry, target = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
                subject = '0' # str(subject.item())
                #if features.shape[0] != batch_size:
                #    continue

                # Zero gradients
                optimizer.zero_grad()

                # Model pass
                output, _, loss_dict = model.train_step((features, captions, hidden, carry), target, subject=subject)  # (bs, max_len, vocab_size)

                # L2 loss - encoder
                #l2_loss_enc = L2_loss(model.parameters(), 0.01)

                # Backprop
                loss = loss_dict['loss']
                total_loss = loss #+ l2_loss_enc  # Add encoder l2

                loss_ = loss.detach().item()
                acc_  = loss_dict['accuracy'].detach().item()
                l2_loss_enc_ = 0 # l2_loss_enc.detach().item()

                epoch_loss['val_loss'].append(loss_)
                epoch_loss['val_accuracy'].append(acc_)
                epoch_loss['val_L2'].append(l2_loss_enc_)
                losses.append({'epoch': epoch, 'batch': batch_nr, 'subject': subject, 'val_loss': loss_, 'enc-L2': l2_loss_enc_, 'val_accuracy': acc_})

            print(f"validation: {epoch:02}/{epochs} | loss: {np.mean(epoch_loss['val_loss']):.3f} | enc-L2: {np.mean(epoch_loss['val_L2']):.3f} | accuracy: {np.mean(epoch_loss['val_accuracy']):.3f}")


        # --- Post epoch ---
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'{run_path}/model/model_ep{epoch:03}.pt')
        pd.DataFrame(losses).to_csv(f"{run_path}/loss_history.csv", mode='w', header=not os.path.exists(f"{run_path}/loss_history.csv"))  # use mode='a' if appending
    # --- Post training ---
    print("Training Complete!")
    return





def load_model(model, path):
    """ Load model checkpoint """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    try:
        loss  = checkpoint['loss']
    except KeyError:
        loss = np.nan
    return model, epoch, loss


def generate_predictions_batch(model, epoch:int, generator):
    """ Generate predictions from """

    outputs = defaultdict(list)
    attention_maps = defaultdict(list)

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for batch_nr, data in enumerate(generator):
            features, captions, hidden, carry, subject, target = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5]
            subject = str(subject.item())

            # Start word
            start_seq = np.repeat([tokenizer.word_index['<start>']], features.shape[0])
            start_seq = torch.from_numpy(start_seq).to(device)

            output, attn_map = model.inference_step(features, start_seq, hidden, carry, subject, max_len)  # [bs, max_len], [bs, max_len, regions, 1]

            outputs[subject].append(output.cpu().numpy())
            attention_maps[subject].append(attn_map.cpu().numpy())

    output = []
    for (k,v) in outputs.items():
        temp = [i for sublist in v for i in sublist]
        temp = np.stack(temp, axis=0)  # (515, 15)
        output.append(temp)
    output = np.stack(output, axis=0)
    print("output:", output.shape)

    attention_map = []
    for (k,v) in attention_maps.items():
        temp = [i for sublist in v for i in sublist]
        temp = np.stack(temp, axis=0)
        attention_map.append(temp)
    attention_map = np.stack(attention_map, axis=0)
    print("attenion_map:", attention_map.shape)

    add_name = ""
    np.save(open(f"{out_path}/output_captions_{epoch}{add_name}.npy", "wb"), output)
    np.save(open(f"{out_path}/attention_scores_{epoch}{add_name}.npy", "wb"), attention_map)
    with open(f"{out_path}/tokenizer.json", "w") as f:
        f.write(tokenizer.to_json())

    print(f"Inference complete! ({(time.time() - start_time):.1f}s)")

    return

def generate_predictions(model, epoch: int, generators: dict):
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

    start_time = time.time()
    outputs = []
    attention_maps = []

    inf_subjects = list(generators.keys())
    print("inf_subjects:", inf_subjects)
    for sub in inf_subjects:

        encoder_idx = inf_subjects.index(sub)

        sub_outputs = []
        sub_attention_maps = []
        generator = generators[str(sub)]
        print(f"Inference // subject: {sub}")

        model.eval()
        for batch_nr, data in enumerate(generator):

            features, captions, hidden, carry, target = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4]

            # Start word
            start_seq = np.repeat([tokenizer.word_index['<start>']], features.shape[0])
            start_seq = torch.from_numpy(start_seq).to(device)

            output, attn_map = model.inference_step(features, start_seq, hidden, carry, encoder_idx, max_len)  # [bs, max_len], [bs, max_len, regions, 1]
            sub_outputs.append(output)
            sub_attention_maps.append(attn_map)

        outputs.append(sub_outputs)
        attention_maps.append(sub_attention_maps)

    outputs = [torch.cat(o, dim=0) for o in outputs]  # 8 x (samples, max_len)
    outputs = torch.stack(outputs, dim=0).unsqueeze(-1).numpy()  # [8, samples, max_len, 1]
    print("outputs:", outputs.shape)

    attention_maps = [torch.cat(a, dim=0) for a in attention_maps]
    attention_maps = torch.stack(attention_maps, dim=0).numpy()  # [8, samples, max_len, regions, 1]
    print("attention maps:", attention_maps.shape)

    # Save outputs
    add_name = ""
    np.save(open(f"{out_path}/output_captions_{epoch}{add_name}.npy", "wb"), outputs)
    np.save(open(f"{out_path}/attention_scores_{epoch}{add_name}.npy", "wb"), attention_maps)
    with open(f"{out_path}/tokenizer.json", "w") as f:
        f.write(tokenizer.to_json())

    print(f"Inference complete! ({(time.time() - start_time):.1f}s)")
    return


def inference(model, path, generators: dict):
    """ Loads a model and runs inference on the given generators """
    # Load model
    model, epoch, _ = load_model(model, path)
    print("> Model loaded <")
    # Run inference
    generate_predictions_batch(model, epoch, generators)
    return


if __name__ == '__main__':

    if training:
        train(model, optimizer, criterion)
        print("Done.")
    else:
        print("Running inference")
        #inf_model = "/home/hpcgies1/rds/hpc-work/NIC/Log/multi_subject_torch4/model/model_ep009.pt" # 001, 009, 029
        #inf_model = "/home/hpcgies1/rds/hpc-work/NIC/Log/multi_subject_torch_3subs/model/model_ep014.pt"
        #inf_model = "/home/hpcgies1/rds/hpc-work/NIC/Log/torch_alt_batches/model/model_ep008.pt"
        inf_model = "/home/hpcgies1/rds/hpc-work/NIC/Log/multi_head_attention/model/model_ep003.pt"
        inference(model, inf_model, test_generator)
        #inference(model, inf_model, test_generators)
