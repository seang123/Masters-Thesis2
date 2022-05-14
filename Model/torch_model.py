import torch
import torch.nn as nn
import numpy as np

class LocallyConnected(nn.Module):

    def __init__(self, groups, embedding_dim):
        super(LocallyConnected, self).__init__()

        in_groups, out_groups = groups
        self.in_groups = in_groups
        assert out_groups[0] == embedding_dim, f"Embedding shapes miss-aligned {out_groups[0]} != {embedding_dim}"

        self.norm = nn.LayerNorm(embedding_dim)
        self.act  = nn.LeakyReLU(0.2)

        self.layers = nn.ModuleList()
        for i in range(len(in_groups)):
            self.layers.append(nn.Linear(in_groups[i].shape[0], embedding_dim))

    def forward(self, x):
        """ Forward pass """

        # Loop through the layers, take the relevant indices from x as input
        output = []
        for i, l in enumerate(self.layers):
            temp = self.in_groups[i]
            x_i = x[:,self.in_groups[i]]
            out = self.norm(self.act(l(x_i))) # (bs, 32)
            output.append( out )

        return torch.stack(output, dim=1) # (bs, 360, 32)

class NIC(nn.Module):

    def __init__(self, groups, embedding_dim_feat, embedding_dim_text, units, max_len, vocab_size, n_subjects=8 ):
        """
        Parameters:
        -----------
            groups : tuple(list,list(int)) - group indices and size
            embedding_dim : int - size of the encoder embeddings
            units : int - LSTM units
            max_len : int - maximum caption length
            vocab_size : int - size of the vocabulary (output)
        """
        super(NIC, self).__init__()

        in_groups, out_groups = groups
        print("in_groups:", len(in_groups))
        print("out_groups:", len(out_groups))

        self.encoders = nn.ModuleList()
        for i in range(n_subjects):
            self.encoders.append(LocallyConnected(groups, embedding_dim_feat))

        self.emb = nn.Embedding(vocab_size, embedding_dim_text)
        self.lstm= nn.LSTM(
                input_size=embedding_dim_feat + embedding_dim_text,
                hidden_size=units,
                num_layers=1,#max_len,
                batch_first=True)
        self.decoder = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, vocab_size),
                nn.Softmax(dim=-1)
                )

        print("Model initalised")


    def forward(self, x, subject):
        """
        Parameters:
        -----------
            x - raw betas
            subject - int
        """

        features, text, a, c = x
        a = a.unsqueeze(0)
        c = c.unsqueeze(0)

        # Select the right encoder
        encoder  = self.encoders[subject]

        # Encode features
        features =  encoder(features) # (bs, 360, 32)

        # Embed text
        text = self.emb(text)

        output = []
        for i in range(text.shape[1]):
            context = torch.mean(features, dim=1) # TODO attention

            context = torch.cat((context, text[:,i,:]), axis=1)
            context = context.unsqueeze(1)

            _, (a, c) = self.lstm(context, (a, c))

            out = self.decoder(a)

            output.append( out )

        output = torch.stack(output, dim=0) # (15, 1, bs, 5001)
        output = torch.swapaxes(output[:,0,:,:], 0, 1) # (bs, 15, 5001)
        return output

