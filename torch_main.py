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
from DataLoaders.torch_generator import Dataset, Dataset_mix, Dataset_batch
from torchinfo import summary
import argparse
from rich.console import Console
console = Console()

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


@loader.timeit
def testing():
    train_pairs = []
    val_pairs = []
    test_pairs = []
    for sub in range(8):
        t_ = []
        for i in range(90):
            t_.append([sub, f"<start> hello this is a test <end>", sub, i**2])
        train_pairs.append(t_)
    for sub in range(8):
        t_ = []
        for i in range(48):
            t_.append([sub, f"bye this is a test", sub, i**2])
        val_pairs.append(t_)
    for sub in range(8):
        t_ = []
        for i in range(51):
            t_.append([sub, f"hello and goodbye, test", sub, i * 2])
        test_pairs.append(t_)

    train_pairs = np.array(train_pairs)
    val_pairs = np.array(val_pairs)
    test_pairs = np.array(test_pairs)
    train_betas = {}
    val_betas = {}
    test_betas = {}
    for sub in range(1, 9):
        train_betas[str(sub)] = np.random.uniform(0, 1, (90, 327684))
        val_betas[str(sub)] = np.random.uniform(0, 1, (48, 327684))
        test_betas[str(sub)] = np.random.uniform(0, 1, (51, 327684))
    return train_pairs, val_pairs, test_pairs, train_betas, val_betas, test_betas

@loader.timeit
def testing2():
    train_pairs = []
    val_pairs   = []
    test_pairs  = []
    for sub in range(1, 9):
        t, v, te = loader.get_nsd_keys(str(sub))
        train_pairs.append(loader.create_pairs(t, str(sub))[:190])
        if sub != 4 and sub != 8:
            val_pairs.append(loader.create_pairs(v, str(sub))[:148])
        test_pairs.append(loader.create_pairs(te, str(sub))[:151])

    train_pairs = np.array(train_pairs)
    val_pairs = np.array(val_pairs)
    test_pairs = np.array(test_pairs)
    train_betas = {}
    val_betas = {}
    test_betas = {}
    for sub in range(1, 9):
        train_betas[str(sub)] = np.random.uniform(0, 1, (190, 327684))
        val_betas[str(sub)] = np.random.uniform(0, 1, (148, 327684))
        test_betas[str(sub)] = np.random.uniform(0, 1, (151, 327684))
    return train_pairs, val_pairs, test_pairs, train_betas, val_betas, test_betas


print("------ Data ------")


def init_dataset_mix(train_subs: list):
    """ Generated a mixed dataset """
    train_pairs, val_pairs, test_pairs, train_betas, val_betas, test_betas = loader.load_subs(train_subs)  # (subs 4 and 8 val set is == test set)
    #train_pairs = np.array(train_pairs)
    #val_pairs   = np.array(val_pairs)
    #test_pairs  = np.array(test_pairs)

    #train_pairs, val_pairs, test_pairs, train_betas, val_betas, test_betas = testing2()  # train_pairs: [subs, samples, 4]

    def batchify(pairs: list, batch_size: int):
        # [subs, samples, 4]
        batches = []
        #for p in range(pairs.shape[0]):
        for p in range(len(pairs)):
            b = []
            for i in range(0, len(pairs[p]), batch_size):
                b.append(pairs[p][i:i + batch_size])
            batches.append(b)
        return batches

    train_pairs = batchify(train_pairs, batch_size) #  [subs, batches, batch_size, 4]
    val_pairs   = batchify(val_pairs, batch_size)
    test_pairs  = batchify(test_pairs, batch_size)
    print("train_pairs:", len(train_pairs), "// train_pairs[0]:", len(train_pairs[0]), "// train_pairs[0][0]:", len(train_pairs[0][0]))

    # Flatten the pairs
    train_pairs = [i for sublist in train_pairs for i in sublist]  # flatten the pairs
    val_pairs   = [i for sublist in val_pairs for i in sublist]  # flatten the pairs
    test_pairs  = [i for sublist in test_pairs for i in sublist]  # flatten the pairs

    # Init Generators
    train_dataset = Dataset_batch(train_pairs, train_betas, tokenizer, config['units'], config['max_length'], vocab_size, batch_size, device)
    train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=0, pin_memory=True)

    val_dataset = Dataset_batch(val_pairs, val_betas, tokenizer, config['units'], config['max_length'], vocab_size, batch_size, device)
    val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=0, pin_memory=True)

    test_dataset = Dataset_batch(test_pairs, test_betas, tokenizer, config['units'], config['max_length'], vocab_size, batch_size, device)
    test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=None, shuffle=False, num_workers=0, pin_memory=True)

    return train_generator, val_generator, test_generator


def init_dataset(train_subs: list = [1, 2, 3, 4, 5, 6, 7, 8], val_subs: list = [1, 2, 3, 5, 6, 7], get_test=True):
    """  Load the training/validation/testing data

    val_subs should be a subset of train_subs

    Returns:
        Data Generators inside dict
    """
    train_pairs, val_pairs, test_pairs, train_betas, val_betas, test_betas = loader.load_subs(train_subs)  # (subs 4 and 8 val set is == test set)

    # Synthetic data (for testing)
    #train_pairs, val_pairs, test_pairs, train_betas, val_betas, test_betas = testing()

    n_subjects = len(train_subs)

    train_generators = {}
    val_generators   = {}
    test_generators  = {}

    train_subs = list(map(str, train_subs))
    for k, i in enumerate(train_subs):
        train_dataset = Dataset(train_pairs[int(k)], train_betas[i], tokenizer, config['units'], config['max_length'], vocab_size, device)
        train_generators[i] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    if len(val_subs) > 0:
        val_subjects = list(map(str, val_subs))  # sub 4 and 8 have no val set
        for k, i in enumerate(val_subjects):
            val_dataset = Dataset(val_pairs[int(k)], val_betas[i], tokenizer, config['units'], config['max_length'], vocab_size, device)
            val_generators[i] = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    if get_test:
        for k, i in enumerate(train_subs):
            test_dataset = Dataset(test_pairs[int(k)], test_betas[i], tokenizer, config['units'], config['max_length'], vocab_size, device)
            test_generators[i] = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_generators, val_generators, test_generators, n_subjects


train_subs = [1, 2, 3, 4, 5, 6, 7, 8]  # [1, 2, 3, 4, 5, 6, 7, 8]
val_subs   = [1, 2, 3, 5, 6, 7]
n_subjects = len(train_subs)
"""
if training:
    train_generators, val_generators, _, n_subjects = init_dataset(train_subs, val_subs)
else:
    train_generators, _, test_generators, n_subjects = init_dataset(train_subs, val_subs)

print(">> N subjects:", n_subjects, "<<")
"""
train_generator, val_generator, test_generator = init_dataset_mix(train_subs)

#
# ------------- Model ------------
#
model = NIC(groups, 32, 512, 512, config['max_length'], vocab_size, subjects=[1, 2, 3, 4, 5, 6, 7, 8]).to(device)  # n_subjects=n_subjects).to(device)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), weight_decay=3.0e-5)  # 0.0003
#for k, v in enumerate(train_generators['1']):
#    print(summary(model, input_data={'x':v[:-1], 'subject':1}, batch_dim=64, verbose=0))


# Initialise the model with one batch of data
if training==False:
    with torch.no_grad():
        """
        for sub in train_subs:
            generator = train_generators[str(sub)]
            encoder_idx = train_subs.index(sub)
        """
        for batch_nr, data in enumerate(train_generator):
            features, captions, hidden, carry, subject, target = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device)
            subject = str(subject.item())

            # Model pass
            output, _, loss_dict = model.train_step((features, captions, hidden, carry), target, subject=subject)  # (bs, max_len, vocab_size)
            break
            print(f"Model initalised on sub: {sub}")


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


def train_mix(model, optimizer, criterion):
    """ Trains on alternating batches """

    losses = []
    for epoch in range(1, epochs + 1):
        print(f" ---== Epoch :: {epoch} ==---")
        epoch_loss = defaultdict(list)

        model.train()
        for batch_nr, data in enumerate(train_generator):
            batch_time = time.time()
            features, captions, hidden, carry, subject, target = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device)
            subject = str(subject.item())
            #if features.shape[0] != batch_size:
            #    continue

            # DataLoader time
            prepare_time = batch_time - time.time()

            # Zero gradients
            optimizer.zero_grad()

            # Model pass
            output, _, loss_dict = model.train_step((features, captions, hidden, carry), target, subject=subject)  # (bs, max_len, vocab_size)

            # Model forward pass time
            model_time = batch_time - prepare_time - time.time()

            # L2 loss - encoder
            l2_loss_enc = L2_loss(model.parameters(), 0.01)

            # Backprop
            loss = loss_dict['loss']
            total_loss = loss + l2_loss_enc  # Add encoder l2
            total_loss.backward()
            optimizer.step()

            # Store loss data
            loss_ = loss.detach().item()
            acc_  = loss_dict['accuracy'].detach().item()
            l2_loss_enc_ = l2_loss_enc.detach().item()

            epoch_loss['loss'].append(loss_)
            epoch_loss['accuracy'].append(acc_)
            epoch_loss['L2'].append(l2_loss_enc_)
            losses.append({'epoch': epoch, 'batch': batch_nr, 'subject': subject, 'loss': loss_, 'enc-L2': l2_loss_enc_, 'accuracy': acc_})

            process_time = batch_time - time.time() - prepare_time
            # Print information
            if batch_nr % 25 == 0:
                print(f"epoch: {epoch:02}/{epochs} | {batch_nr:03} | loss: {np.mean(epoch_loss['loss']):.3f} | enc-L2: {np.mean(epoch_loss['L2']):.3f} | acc: {np.mean(epoch_loss['accuracy']):.3f} | comp. eff.: {(process_time/(process_time+prepare_time)):.3f}")
        print(f"Epoch: {epoch} complete!")

        # Evaluation
        # ----------
        model.eval()
        with torch.no_grad():
            for batch_nr, data in enumerate(val_generator):
                features, captions, hidden, carry, subject, target = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device)
                subject = str(subject.item())
                #if features.shape[0] != batch_size:
                #    continue

                # Zero gradients
                optimizer.zero_grad()

                # Model pass
                output, _, loss_dict = model.train_step((features, captions, hidden, carry), target, subject=subject)  # (bs, max_len, vocab_size)

                # L2 loss - encoder
                l2_loss_enc = L2_loss(model.parameters(), 0.01)

                # Backprop
                loss = loss_dict['loss']
                total_loss = loss + l2_loss_enc  # Add encoder l2

                loss_ = loss.detach().item()
                acc_  = loss_dict['accuracy'].detach().item()
                l2_loss_enc_ = l2_loss_enc.detach().item()

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

def train(model, optimizer, criterion):

    #losses = defaultdict(list)
    for epoch in range(1, epochs + 1):
        print(f" ---== Epoch :: {epoch} ==---")

        losses = []
        epoch_time = time.time()
        for sub, sub_id in enumerate(train_subs):

            encoder_idx = train_subs.index(sub_id)
            generator = train_generators[str(sub_id)]
            sub_loss = defaultdict(list)
            sub_start_time = time.time()

            model.train()
            # Loop through the dataset
            for batch_nr, data in enumerate(generator):
                batch_time = time.time()
                # Access data
                features, captions, hidden, carry, target = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)

                # DataLoader time
                prepare_time = batch_time - time.time()

                # Zero gradients
                optimizer.zero_grad()

                # Model pass
                output, _, loss_dict = model.train_step((features, captions, hidden, carry), target, subject=encoder_idx)  # (bs, max_len, vocab_size)

                # Model forward pass time
                model_time = batch_time - prepare_time - time.time()

                l2_loss_enc = L2_loss(model.parameters(), 0.01)

                # Backprop
                loss = loss_dict['loss']
                loss += l2_loss_enc  # Add encoder l2
                loss.backward()
                optimizer.step()

                # Store loss
                loss_ = loss_dict['loss'].detach().item()
                acc_  = loss_dict['accuracy'].detach().item()
                sub_loss['sub_epoch_loss'].append(loss_)
                sub_loss['sub_epoch_acc'].append(acc_)
                losses.append({'epoch': epoch, 'sub': sub_id, 'batch': batch_nr, 'loss': loss_, 'accuracy': acc_})

                process_time = batch_time - time.time() - prepare_time
                # Print information
                if batch_nr % 25 == 0:
                    print(f"epoch: {epoch:02}/{epochs} | sub: {sub_id} | {batch_nr:03} | loss: {np.mean(sub_loss['sub_epoch_loss']):.3f} | acc: {np.mean(sub_loss['sub_epoch_acc']):.3f} | comp. eff.: {(process_time/(process_time+prepare_time)):.3f}")

            # Store loss
            print(f"--- sub: {sub} // time: {(time.time()-sub_start_time):.2f}s // loss: {np.mean(sub_loss['sub_epoch_loss']):.3f} ---")

        # ----------
        # Validation
        # ----------
        model.eval()
        with torch.no_grad():

            for _, v_sub_id in enumerate(val_subs):
                val_loss = defaultdict(list)
                val_generator = val_generators[str(v_sub_id)]
                encoder_idx = val_subs.index(v_sub_id)

                for batch_nr, data in enumerate(val_generator):

                    # Data prep
                    features, captions, hidden, carry, target = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)

                    # Model pass
                    output, _, loss_dict = model.train_step((features, captions, hidden, carry), target, subject=encoder_idx)  # (bs, max_len, vocab_size)

                    # Losses
                    loss_ = loss_dict['loss'].detach().item()
                    acc_  = loss_dict['accuracy'].detach().item()
                    val_loss['loss'].append(loss_)
                    val_loss['accuracy'].append(acc_)
                    losses.append({'epoch': epoch, 'sub': v_sub_id, 'batch': batch_nr, 'val_loss': loss_, 'val_accuracy': acc_})

                print(f"validation: {epoch:02}/{epochs} | sub: {v_sub_id} | loss: {np.mean(val_loss['loss']):.3f} | accuracy: {np.mean(val_loss['accuracy']):.3f}")

        ## ----- Post epoch -----

        # Store Loss
        pd.DataFrame(losses).to_csv(f"{run_path}/loss_history.csv", mode='a', header=not os.path.exists(f"{run_path}/loss_history.csv"))
        # Save model
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'{run_path}/model/model_ep{epoch:03}.pt')

        print(f"epoch: {epoch:02}/{epochs} complete! ({(time.time()-epoch_time):.1f}s)\n")


    # ---- Post training ----
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

            outputs[subject].append(output.numpy())
            attention_maps[subject].append(attn_map.numpy())

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
        #train(model, optimizer, criterion)
        train_mix(model, optimizer, criterion)
        print("Done.")
    else:
        print("Running inference")
        #inf_model = "/home/hpcgies1/rds/hpc-work/NIC/Log/multi_subject_torch4/model/model_ep009.pt" # 001, 009, 029
        #inf_model = "/home/hpcgies1/rds/hpc-work/NIC/Log/multi_subject_torch_3subs/model/model_ep014.pt"
        inf_model = "/home/hpcgies1/rds/hpc-work/NIC/Log/torch_alt_batches/model/model_ep008.pt"
        inference(model, inf_model, test_generator)
        #inference(model, inf_model, test_generators)
