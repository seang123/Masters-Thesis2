import logging
import time
import pandas as pd
import os, sys
import csv
import tensorflow as tf
from tensorflow.keras.optimizers import schedules
import tensorflow_addons as tfa
from tensorflow.keras.utils import Progbar
import numpy as np
from Model import lc_NIC
from DataLoaders import load_avg_betas2 as loader
from DataLoaders import data_generator as generator
from Callbacks import BatchLoss, EpochLoss, WarmupScheduler, Predict
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from collections import defaultdict
from datetime import datetime
import subprocess
import yaml

gpu_to_use = 0
print(f"Running on GPU: {gpu_to_use}", flush=True)

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

logging.basicConfig(filename=f'{run_path}/log.log', filemode='w', level=logging.DEBUG)
logging.info(f"training starts at:   {datetime.now()}")

np.random.seed(config['seed'])
tf.random.set_seed(config['seed'])

# Copy Model file to run_path folder for record
subprocess.run(["cp", "-r", "/home/hpcgies1/Masters-Thesis/AttemptFour/Model", f"{run_path}/Model_files"], shell=False, check=True)
print(f"Model file copied to {run_path} for record", flush=True)

## Parameters
vocab_size = config['top_k'] + 1

subject = '5'
print(">> subject: ", subject, " <<")
#
## Load data
#
train_keys, val_keys, test_keys = loader.get_nsd_keys(subject)
print("train_keys:", train_keys[:10])
print("val_keys:", val_keys[:10])
print("train_keys:", train_keys.shape)
print("val_keys:", val_keys.shape)
print("test_keys:", test_keys.shape)

# Create pairs
train_pairs = np.array(loader.create_pairs(train_keys, subj=subject))
val_pairs = np.array(loader.create_pairs(val_keys, subj=subject))
print("train_pairs:", train_pairs.shape)
print("val_pairs:  ", val_pairs.shape)

# Load Betas
train_betas, val_betas, _ = loader.load_split_betas(subject)
print("train_betas:", train_betas.shape)
print("val_betas:", val_betas.shape)

#tokenizer = loader.load_tokenizer()
tokenizer, _ = loader.build_tokenizer(np.arange(1, 73001), config['top_k'])

# Setup optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1 = 0.9, beta_2=0.98, epsilon=10.0e-9, clipnorm=config['clipnorm'])
print(f"Using optimizer: Adam", flush=True)

# Loss function
loss_object = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        reduction='none'
)

# Setup Model
model = lc_NIC.NIC(
        loader.get_groups(config['group_size'], separate_hemi=True),
        #loader.select_groups(config['group_size'], remove=[142,17,133,315,1, 197,158,192,135,153,137,140,92,183,125]),
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
#Compile
model.compile(optimizer, loss_object, run_eagerly=True)

## The following relates to pre-loading LSTM weights
init_generator = generator.DataGenerator(
        train_pairs,
        train_betas,
        config['batch_size'],
        tokenizer,
        config['units'],
        config['max_length'],
        vocab_size,
        pre_load_betas=False,
        shuffle=False, training=True)
build_time = time.perf_counter()
temp = init_generator.__getitem__(0)[0]
print(len(temp))
print(temp[0].shape)
model(temp, training=False)
print(f"Model build time: {(time.perf_counter() - build_time):.3f}", flush=True)
print(model.summary(), flush=True)


# Setup Checkpoint handler
checkpoint_path = f"{run_path}/model/"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path_best = f"{checkpoint_path}" + "model-ep{epoch:03d}.h5"
checkpoint_best = ModelCheckpoint(
        checkpoint_path_best,
        monitor='val_loss',
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
        mode='min',
        period=1
)

#
## Callbacks
#
loss_history = EpochLoss.LossHistory(f"{run_path}/loss_history.csv", f"{run_path}")

logdir = f"./tb_logs/scalars/{config['run']}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
tensorboard_callback = TensorBoard(
        log_dir=logdir,
        update_freq='batch')

#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
#        lr_schedule, verbose = 0)

_callbacks = [
        loss_history,
        #tensorboard_callback,
        checkpoint_best,
]
callbacks = tf.keras.callbacks.CallbackList(
            _callbacks, add_history=True, model=model)

logs = {}
start_epoch = 0

def dotfit():
    logging.info("training with .fit()")

    train_generator = generator.DataGenerator(
            train_pairs,
            train_betas,
            config['batch_size'],
            tokenizer,
            config['units'],
            config['max_length'],
            vocab_size,
            pre_load_betas=False,
            shuffle=True, training=True)
    val_generator = generator.DataGenerator(
            val_pairs,
            val_betas,
            config['batch_size'],
            tokenizer,
            config['units'],
            config['max_length'],
            vocab_size,
            pre_load_betas=False,
            shuffle=False, training=True)

    model.fit(
            train_generator,
            epochs = config['epochs'],
            steps_per_epoch = len(train_pairs)//config['batch_size'],
            batch_size = config['batch_size'],
            callbacks = _callbacks,
            validation_data = val_generator,
            validation_steps = len(val_pairs)//config['batch_size'],
            initial_epoch = start_epoch)
    return




def custom_train_loop():
    print(f"------\nRunning custom training loop")
    print(f"for {config['epochs'] - start_epoch} epochs\n------")
    logging.info("training with custom training loop")

    train_generator = generator.DataGenerator(train_pairs, config['batch_size'], tokenizer, config['units'], config['max_length'], vocab_size, shuffle=True, training=True)
    val_generator = generator.DataGenerator(val_pairs, config['batch_size'], tokenizer, config['units'], config['max_length'], vocab_size, shuffle=True, training=True)

    grads = []

    # Train for N epochs
    callbacks.on_train_begin(logs=logs)
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nepoch {epoch+1}/{config['epochs']}")
        epoch_start_time = time.time()
        callbacks.on_epoch_begin(epoch, logs=logs)

        # Reshuffle train/val pairs
        #train_pairs = loader.create_pairs(train_keys, config['dataset']['captions_path'], seed = shuffle_seed)
        #val_pairs   = loader.create_pairs(val_keys,   config['dataset']['captions_path'], seed = shuffle_seed)
        # Instantiate new generator
        #train_generator = create_generator(train_pairs, True)
        #val_generator = create_generator(val_pairs, False)

        #batch_train_loss = defaultdict(list)
        #batch_val_loss = defaultdict(list)

        # Progress bar
        pb = Progbar(len(train_pairs)/config['batch_size'])#, stateful_metrics=['loss', 'l2', 'accuracy'])
        pb2 = Progbar(len(val_pairs)/config['batch_size'])#, stateful_metrics=['val-loss', 'val-l2', 'val-accuracy'])

        # Training
        for (batch_nr, data) in enumerate(train_generator):
            callbacks.on_batch_begin(epoch, logs=logs)
            #target = data[1]
            #target = tokenizer.sequences_to_texts(np.argmax(target, axis=2))

            # data -> ([betas, cap_vector, a0, c0], target)
            #print( "tf.executing_eagerly()", tf.executing_eagerly() )
            losses, grad = model.train_step(data)

            grads.append(grad)

            #for key, v in losses.items():
            #    batch_train_loss[key].append(v)

            values = list(losses.items())
            pb.add(1, values=values)

            callbacks.on_train_batch_end(batch_nr, logs=losses)


        # Validation
        for (batch_nr, data) in enumerate(val_generator):

            losses_val = model.test_step(data)

            #for key, v in losses.items():
            #    batch_val_loss[key].append(v)

            values = list(losses_val.items())
            pb2.add(1, values=values)

            callbacks.on_test_batch_end(batch_nr, logs=losses_val)

        # On-Epoch-End
        callbacks.on_epoch_end(epoch, logs=logs)
        #model.save_weights(f"{config['log']}/{config['run']}/model/checkpoints/checkpoint_latest")

    # On-Train-End
    callbacks.on_train_end(logs=logs)

    df = pd.DataFrame(grads)
    df.to_csv(f'{run_path}/df_grads.csv')
    df.to_pickle(f'{run_path}/df_grads.csv')

    return

if __name__ == '__main__':
    try:
        #custom_train_loop()
        dotfit()
    except KeyboardInterrupt as e:
        print("--Keyboard Interrupt--", flush=True)
    finally:
        print(f"Done.", flush=True)

