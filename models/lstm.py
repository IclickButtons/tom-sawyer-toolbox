import tensorflow as tf 
from tom_sawyer_toolbox.models.rnn_base_model import RNNBaseModel

class LSTM(RNNBaseModel): 
    def __init__(self, 
                 num_time_steps,
                 num_units, 
                 **kwargs):

        self._num_time_steps = num_time_steps
        self._num_units = num_units
        self._D = 1 
        
        super(LSTM, self).__init__(**kwargs) 


    def network(self): 
        self.x_train = tf.placeholder(tf.float32, shape=[64,
        10, self._D])  
        self.y_train = tf.placeholder(tf.float32, shape=[64, 1])  
        layer1 = tf.keras.layers.LSTM(self._num_units, 
                                     time_major=False,
                                     stateful=True,
                                     return_sequences = True, 
                                     batch_input_shape=(self._batch_size,
                                                        self._num_time_steps, 
                                                        1)) 
        layer2 = tf.keras.layers.LSTM(self._num_units, 
                                     time_major=False,
                                     stateful=True) 

        layer1_out = layer1(self.x_train) 
        layer2_out = layer2(layer1_out) 
        y_pred = tf.layers.dense(layer2_out, 1, activation=None,
                                 kernel_initializer=tf.orthogonal_initializer()) 

        return y_pred

    def calculate_loss(self): 
        
        y_pred = self.network() 
        self.loss = tf.reduce_mean(tf.square(y_pred - self.y_train), name='loss')  

        return self.loss 


