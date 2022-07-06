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

        self.layers = nn.ModuleList()
        for i in range(len(in_groups)):
            self.layers.append(nn.Linear(in_groups[i].shape[0], embedding_dim))

    def forward(self, x):
        """ Forward pass """
        # Loop through the layers, take the relevant indices from x as input
        output = [F.dropout(F.leaky_relu(self.norm(layer(
            x[:, self.in_groups[i]])), 0.2), 0.2) for (i, layer) in enumerate(self.layers)]
        return torch.stack(output, dim=1)  # (bs, 360, 32)



class MultiheadAttention(nn.Module):
    """
    Multi-head attention
    Each brain regions has the previous hidden state concatenated onto it
    and can then attend to all other brain regions
    """
    def __init__(self, embedding_dim=544, heads=8):
        super(MultiheadAttention, self).__init__()
        self.multi_head = nn.MultiheadAttention(embedding_dim, heads, batch_first=True)

    def forward(self, hidden, features):
        """
        Parameters:
        -----------
            hidden   - [1, bs, units]
            features - [bs, 360, dim]

        Returns:
        --------
            context      - [bs, 1, 32]
            attn_weights - [bs, 360, 1]
        """
        hidden = torch.swapaxes(hidden, 0, 1)  # out: [bs, 1, 512]
        hidden_repeat = torch.repeat_interleave(hidden, 360, dim=1)  # [bs, 1, 512] -> [bs, 360, 512]

        # 1. Combine the hidden state with each feature vector
        combined = torch.cat((features, hidden_repeat), dim=-1)  # [bs, 360, 544]

        # 2. Multihead attention
        attn_output, attn_output_weights = self.multi_head(combined, combined, combined)  # [bs, 360, embedding_dim], [bs, 360, 360]

        attn_weights = torch.mean(attn_output_weights, axis=1)  # [bs, 360]
        attn_weights = torch.unsqueeze(attn_weights, -1)  # [bs, 360, 1]

        # 3. Apply scaling
        context = torch.sum(features * attn_weights, dim=1)  # [bs, 1, 32]

        return context, attn_weights


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

        hidden_with_time_axis = torch.swapaxes(hidden, 0, 1)  # out: [bs, 1, 512]

        attention_hidden_layer = torch.tanh(
            F.leaky_relu(self.W1(features), 0.2) + F.leaky_relu(self.W2(hidden_with_time_axis), 0.2))  # out: [bs, 360, 32]

        score = self.V(attention_hidden_layer)  # out: [bs, 360, 1]
        attention_weights = self.softmax(score)
        context_vector = torch.sum(attention_weights * features, dim=1)  # out: [bs, 1, 32]
        return context_vector, attention_weights





class NIC(nn.Module):

    def __init__(self, groups, embedding_dim_feat, embedding_dim_text, units, max_len, vocab_size, subjects):  #, n_subjects=8):
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
        #self.encoders = nn.ModuleList()
        #for i in range(n_subjects):
        #    self.encoders.append(LocallyConnected(groups, embedding_dim_feat))
        self.encoders = nn.ModuleDict([[str(i), LocallyConnected(groups, embedding_dim_feat)] for i in subjects])
        print("Encoder ModuleDict:", self.encoders.keys())

        # Attention Mechanism
        #self.attention = Attention(embedding_dim_feat, units, 32)
        self.attention = MultiheadAttention()

        # Embedding layer
        self.emb = nn.Embedding(vocab_size, embedding_dim_text)
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim_feat + embedding_dim_text,
            hidden_size=units,
            num_layers=1,  #max_len,
            batch_first=True)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(units, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, vocab_size),
            nn.Softmax(dim=-1))

        # Layer norm for LSTM
        self.layer_norm = nn.LayerNorm([units])
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
        features = encoder(features)  # (bs, 360, 32)

        # Embed text
        text = self.emb(text)

        output = []
        attention_scores = []
        for i in range(text.shape[1]):
            # Attention
            context, attention_weights = self.attention(a, features)  # [bs, 32], [bs, 360, 1]
            context = torch.cat((context, text[:, i, :]), axis=1)  # (bs, 1, 32 + 512)
            context = context.unsqueeze(1)

            # LSTM
            _, (a, c) = self.lstm(context, (a, c))  # context should have shape: [bs, dim]
            a = self.layer_norm(a)
            seq = F.leaky_relu(a, 0.2)

            # Decoder
            out = self.decoder(seq)

            output.append(out)
            attention_scores.append(attention_weights)

        output = torch.stack(output, dim=0)  # (15, 1, bs, 5001)
        output = torch.swapaxes(output[:, 0, :, :], 0, 1)  # (bs, 15, 5001)
        attention_scores = torch.stack(attention_scores, 1)
        return output, attention_scores

    def predict(self, features, word, hidden, carry):
        """ Generate single word predictions given features and a prior word
        Parameters:
            features - betas
                [bs, 360, 32]
            word - integer encoded word
                [bs]
            hidden - lstm hidden state
                [1, bs, units]
            carry - lstm carry state
                == hidden
        Returns:
            out - [bs, vocab_size]
            attention_weights - [bs, regions, 1]
            hidden = carry - [1, bs, 512](?)
        """
        # Embed text
        word = self.emb(word)  # (bs, 1, 512)

        # Attention
        context, attention_weights = self.attention(hidden, features)  # out: [bs, 32], [bs, 360, 1]
        context = torch.cat((context, word), axis=1)  # (bs, 1, 32 + 512)
        context = context.unsqueeze(1)

        # LSTM
        _, (hidden, carry) = self.lstm(context, (hidden, carry))
        hidden = self.layer_norm(hidden)
        seq = F.leaky_relu(hidden, 0.2)

        # Decoder
        out = self.decoder(seq)  # out: [1, bs, vocab_size]
        out = out.squeeze(0)

        return out, attention_weights, hidden, carry



    def inference_step(self, features, word, hidden, carry, subject, max_len):
        """ Inference step

        Parameters:
        -----------
            data - tuple
                (features, word, hidden, carry)
            subject - int
                subject id [0, 8)
        Returns:
        --------
            output - tensor
            attention maps - tensor
            loss - dict
        """

        # Select the right encoder
        encoder = self.encoders[subject]
        # Encode features
        features = encoder(features)  # (bs, 360, 32)

        # Have to add dim to hidden state
        hidden = hidden.unsqueeze(0)  # out: (1 bs, 512)
        carry  = carry.unsqueeze(0)

        output = []
        attention_scores = []
        for i in range(max_len):
            # Model call
            out, attn_score, hidden, carry = self.predict(features, word, hidden, carry)

            out = torch.argmax(out, dim=-1)
            attention_scores.append(attn_score.detach())
            output.append(out.detach())

            word = out

        output = torch.stack(output, dim=0)  # out: (max_len, bs)
        output = torch.swapaxes(output, 1, 0)  # out: (bs, max_len)
        attention_scores = torch.stack(attention_scores, 1)  # out: (bs, max_len, regions, 1)
        return output, attention_scores


    def train_step(self, data, target, subject):
        """ Single training step for NIC model

        data - tuple
            (features, caption, hidden, carry)
        target - tensor
            target distribution
        subject - int
            subject index for encoder selection [0, 8)
        """
        output, attention_scores = self(data, subject)  # out: [bs, max_len, vocab_size], [bs, max_len, regions, 1]

        loss = 0
        acc = 0
        max_len = target.shape[1]
        for i in range(max_len):
            loss += self.cross_entropy(output[:, i, :], target[:, i, :])
            acc += self.accuracy(output[:, i, :], target[:, i, :]).float()

        loss /= max_len
        acc  /= max_len

        return output, attention_scores, {'loss': loss, 'accuracy': acc}


    def cross_entropy(self, pred, target):
        """ Compute cross entropy between two distributions """
        return torch.mean(-torch.sum(target * torch.log(pred), dim=1))  # (bs, 5001) -> (64) -> (1)


    def accuracy(self, pred, target):
        """ Accuracy computation """
        target_arg_max = torch.argmax(target, dim=1)
        pred_arg_max   = torch.argmax(pred, dim=1)
        count = torch.sum(pred_arg_max == target_arg_max, dim=0)
        return count / target_arg_max.shape[0]
