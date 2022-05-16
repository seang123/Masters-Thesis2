import torch
import torch.nn as nn
import torch.nn.functional as F
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
            out = F.dropout(self.norm(self.act(l(x_i)))) # (bs, 32)
            output.append( out )

        return torch.stack(output, dim=1) # (bs, 360, 32)





class Attention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, units):
        """
        Parameters:
        -----------
            embedding_dim - size of the encoder output
            hidden_dim    - LSTM hidden state size
            units         - attention weight dim
        """
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.W1 = nn.Linear(embedding_dim, units)
        self.W2 = nn.Linear(hidden_dim, units)
        self.V  = nn.Linear(units, 1)

    def forward(self, hidden, features):
        """
        Parameters:
        -----------
            hidden   - input_shape: [1, bs, units]
            features - input_shape: [bs, 360, embedding_dim]
        """

        hidden_with_time_axis = torch.swapaxes(hidden, 0, 1) # out: [bs, 1, 512]

        attention_hidden_layer = torch.tanh(
                F.relu(self.W1(features)) + F.relu(self.W2(hidden_with_time_axis))) # out: [bs, 360, 32]

        score = self.V(attention_hidden_layer) # out: [bs, 360, 1]
        attention_weights = self.softmax(score)
        context_vector = torch.sum(attention_weights * features, dim=1) # out: [bs, 1, 32]
        return context_vector, attention_weights





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
        print("in_groups:", len(in_groups), "out_groups:", len(out_groups))

        # Encoders
        self.encoders = nn.ModuleList()
        for i in range(n_subjects):
            self.encoders.append(LocallyConnected(groups, embedding_dim_feat))

        # Attention Mechanism
        self.attention = Attention(embedding_dim_feat, units, 32)

        # Embedding layer
        self.emb = nn.Embedding(vocab_size, embedding_dim_text)
        # LSTM
        self.lstm= nn.LSTM(
                input_size=embedding_dim_feat + embedding_dim_text,
                hidden_size=units,
                num_layers=1,#max_len,
                batch_first=True)
        # Decoder
        self.decoder = nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Linear(256, vocab_size),
                nn.Softmax(dim=-1)
                )

        # Layer norm for LSTM
        self.layer_norm = nn.LayerNorm([units])

        # Activation func
        self.leaky_relu = nn.LeakyReLU(0.2)

        print("Model initalised")


    def forward(self, x, subject):
        """
        Parameters:
        -----------
            x - (betas, caption, hidden, carry)
            subject - int [0,8)
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
        attention_scores = []
        for i in range(text.shape[1]):
            #context = torch.mean(features, dim=1) # TODO attention
            context, attention_weights = self.attention(a, features) # [bs, 32], [bs, 360, 1]

            context = torch.cat((context, text[:,i,:]), axis=1) # (bs, 1, 32 + 512)
            context = context.unsqueeze(1)

            # LSTM
            _, (a, c) = self.lstm(context, (a, c))
            a = F.dropout(self.layer_norm(self.leaky_relu(a)))

            # Decoder
            out = self.decoder(a)

            output.append( out )
            attention_scores.append( attention_weights )

        output = torch.stack(output, dim=0) # (15, 1, bs, 5001)
        output = torch.swapaxes(output[:,0,:,:], 0, 1) # (bs, 15, 5001)
        attention_scores = torch.stack(attention_scores, 1)
        return output, attention_scores

    def cross_entropy(self, pred, target):
        """ Compute cross entropy between two distributions """
        return torch.mean(-torch.sum(target * torch.log(pred), dim=1))# (bs, 5001) -> (64) -> (1)

    def accuracy(self, pred, target):
        """ Accuracy computation """
        target_arg_max = torch.argmax(target, dim=1)
        pred_arg_max   = torch.argmax(pred, dim=1)
        count = torch.sum(pred_arg_max == target_arg_max, dim=0)
        return count / target_arg_max.shape[0]


    def train_step(self, data, target, subject):
        """ Single training step for NIC model

        data - tuple
            (features, caption, hidden, carry)
        target - tensor
            target distribution
        subject - int
            subject index for encoder selection [0, 8)
        """
        output, attention_scores = self(data, subject)

        loss = 0
        acc = 0
        max_len = target.shape[1]
        for i in range(max_len):
            loss += self.cross_entropy(output[:,i,:], target[:,i,:])
            acc += self.accuracy(output[:,i,:], target[:,i,:]).float()

        loss /= max_len
        acc  /= max_len

        return output, attention_scores, {'loss':loss, 'accuracy':acc}




