import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

## List of loss directories to plot
loss_dir = {
        0: "subject_2_dense_layer_norm",
        1: "subject_2_lstm_layer_norm",
}

# ['batch_L2' 'batch_accuracy' 'batch_attention' 'batch_loss' 'batch_lr' 'epoch_L2' 'epoch_accuracy' 'epoch_attention' 'epoch_loss' 'epoch_lr']

def get_loss(model_name: str):

    df = pd.read_csv(f"./Log/{model_name}/loss_history.csv")

    ## Example for getting loss or val_loss
    #loss = df.dropna(subset=['loss']).groupby('epoch')['loss'].apply(list)
    #loss = [loss[i][-1] for i in range(len(loss))]
    #val = df.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].apply(list)
    #val = [val[i][-1] for i in range(len(val))]

    return df

def plot_loss(df1):

    fig = plt.figure(figsize=(16,9))

    loss = df1.dropna(subset=['loss']).groupby('epoch')['loss'].apply(list)
    loss = [loss[i][-1] for i in range(len(loss))]
    val = df1.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].apply(list)
    val = [val[i][-1] for i in range(len(val))]
    plt.plot(loss, label='train')
    plt.plot(val, label='val')

    # plot parameters
    #plt.ylim(ymin=1.0)
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Cross-entropy loss")
    plt.savefig(f"./Eval/loss.png")

    plt.close(fig)

def plot_():
    # plot const_lr2 vs multi_sub_sep_enc
    df1 = get_loss(loss_dir[0])
    df2 = get_loss(loss_dir[3])

    fig = plt.figure(figsize=(16,9))

    ## Const lr
    # Train loss
    loss = df1.dropna(subset=['loss']).groupby('epoch')['loss'].apply(list)
    loss = [loss[i][-1] for i in range(len(loss))]
    plt.plot(loss, label='train')
    # val loss
    loss = df1.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].apply(list)
    loss = [loss[i][-1] for i in range(len(loss))]

def plot_2():
    df1 = get_loss(loss_dir[0])

    fig = plt.figure(figsize=(16,9))

    ## Loss
    # val
    loss = df1.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].apply(list)
    loss = [loss[i][-1] for i in range(len(loss))]
    plt.plot(loss, color='darkorange', label='val')
    # Train
    loss = df1.dropna(subset=['loss']).groupby('epoch')['loss'].apply(list)
    loss = [loss[i][-1] for i in range(len(loss))]
    plt.plot(loss, alpha=0.5, color='darkgray', label='train')


    plt.xlim(xmin = -(.05 * plt.gca().get_xlim()[1]))
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Cross-entropy loss")
    plt.savefig(f"./Eval/{loss_dir[0]}_loss.png", bbox_inches='tight')

    plt.close(fig)






if __name__ == '__main__':
#    df = get_loss(loss_dir[3])
#    plot_loss(df)

    plot_2()

