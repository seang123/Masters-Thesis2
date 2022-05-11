import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ELU, ReLU
from tensorflow.keras.initializers import HeNormal, GlorotNormal
from tensorflow.keras.activations import tanh

class Attention(tf.keras.layers.Layer):
    """ True attention as done in Show, Attend, and Tell
    Given the hidden state of the lstm at t_i, and a set of features
    return the input for the LSTM on the next timestep t_i+1
    """

    def __init__(self, units, dropout, **kwargs):
        super(Attention, self).__init__()

        self.softmax = tf.keras.layers.Softmax(axis=1)
        self.dropout = dropout

        #self.bn = tf.keras.layers.BatchNormalization(name = 'attention_bn')
        #self.bn = tf.keras.layers.LayerNormalization(name = 'attention_ln')

        self.W1 = tf.keras.layers.Dense(units, **kwargs)
        self.W2 = tf.keras.layers.Dense(units, **kwargs)
        self.V  = tf.keras.layers.Dense(1)

    def call(self, hidden, features, training=False):
        """ Forward pass """

        hidden_with_time_axis = tf.expand_dims(hidden, 1) # (bs, 1, units)

        attention_hidden_layer = tanh(
                self.W1(features, training=training) +
                self.W2(hidden_with_time_axis, training=training)
        ) # (bs, regions, attn_units)

        #attention_hidden_layer = self.bn(attention_hidden_layer, training=training)
        attention_hidden_layer = self.dropout(attention_hidden_layer, training=training)

        score = self.V(attention_hidden_layer, training=training) # (bs, regions, 1)

        attention_weights = self.softmax(score) # (bs, regions, 1)

        context_vector = tf.reduce_sum(attention_weights * features, axis=1) # (bs, embed_dim)

        return context_vector, attention_weights#, attention_hidden_layer

class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units: int, **kwargs):
        super(LuongAttention, self).__init__()
        self.downscale = tf.keras.layers.Dense(units, **kwargs)

    def call(self, q, v, training=False):
        """ Dot-product attention (luong style)
        Parameters:
        -----------
            q (query) - ndarray
                LSTM hidden state [bs, Tq, dim] (bs, 1, 512)
            v (value) - ndarray
                Encoded neural features [bs, Tv, dim] (bs, 360, 512)
        Return:
        -------
            output - ndarray
                scalled output [bs, Tq, dim]
            attention scores - ndarray
                attention scores [bs, Tq, Tv]
        """

        q = tf.expand_dims(q, 1)
        q = self.downscale(q, training=training) # Downscale hidden state from 512 to 32
        output, attention_scores = tf.keras.layers.Attention()([q, v], return_attention_scores=True, training=training)
        return output, attention_scores
