import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, num_categories, hidden_sizes, activations, mean_xs=None, demean_input=False,
                 final_layer_bias_initializer='zeros', name='encoder'):
        super().__init__(name=name)
        assert len(activations) == len(hidden_sizes)
        
        self.hidden_sizes = hidden_sizes
        self.activations = activations
        self.output_event_shape = (hidden_sizes[-1], num_categories)
        self.network = tf.keras.Sequential()
        
        if demean_input:
            if mean_xs is not None:
                self.network.add(tf.keras.layers.Lambda(lambda x: x - mean_xs))
            else:
                self.network.add(tf.keras.layers.Lambda(lambda x: 2. * tf.cast(x, tf.float32) - 1.))
        for i in range(len(hidden_sizes) - 1):
            self.network.add(tf.keras.layers.Dense(units=hidden_sizes[i], activation=activations[i]))

        self.network.add(tf.keras.layers.Dense(units=hidden_sizes[-1] * num_categories, 
                                               activation=activations[-1],
                                               bias_initializer=final_layer_bias_initializer))
        self.network.add(tf.keras.layers.Reshape(self.output_event_shape))
        
    def call(self, input_tensor):
        return self.network(input_tensor)
    
    
class Decoder(tf.keras.Model):
    def __init__(self, hidden_sizes, activations, mean_xs=None, demean_input=False,
                 final_layer_bias_initializer='zeros', name='decoder'):
        super().__init__(name=name)
        assert len(activations) == len(hidden_sizes)
        
        self.hidden_sizes = hidden_sizes
        self.activations = activations
        self.output_event_shape = hidden_sizes[-1]
        self.network = tf.keras.Sequential()
        
        if demean_input:
            if mean_xs is not None:
                self.network.add(tf.keras.layers.Lambda(lambda x: x - mean_xs))
            else:
                self.network.add(tf.keras.layers.Lambda(lambda x: 2. * tf.cast(x, tf.float32) - 1.))
        for i in range(len(hidden_sizes) - 1):
            self.network.add(tf.keras.layers.Dense(units=hidden_sizes[i], activation=activations[i]))

        self.network.add(tf.keras.layers.Dense(units=hidden_sizes[-1], 
                                               activation=activations[-1],
                                               bias_initializer=final_layer_bias_initializer))
        
    def call(self, input_tensor):
        return self.network(input_tensor)