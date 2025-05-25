import tensorflow as tf
from tensorflow.keras.layers import Layer


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
        )
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time_steps, features)
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1))  # (batch, time_steps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, time_steps, 1)
        context = tf.reduce_sum(inputs * attention_weights, axis=1)  # (batch, features)
        return context, attention_weights
