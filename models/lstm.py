import tensorflow as tf 

class LSTM(RNNBaseModel): 
    def __init__(self, 
                 look_back,
                 num_units, 
                 **kwargs):

        self._look_back = look_back 
        self._num_units = num_units
        super(LSTM, self).__init__(**kwargs) 


    def network(self)
    layer1 = tf.keras.layer.LSTM(self.num_units, time_major=False,
            stateful=True, batch_input_shape=(self._batch_size,
                self._num_time_steps, 1)  
    layer2 = tf.keras.layer.Lstm(self.num_units, time_major=False,
        stateful=True 

    layer1_out = layer1(train_x) 
    layer2_out = layer2(layer1_out) 
    y_pred = tf.layers.dense(layer2_out, 1, activation=None,
        kernel_initializer=tf.orthogonal_initalizer()) 

    return y_pred

    def calculate_loss(self, y_pred, y_true): 
        self.x_train = tf.placeholder(tf.float32, shape=[None, None, self._D] 
        self.y_train = tf.placeholder(tf.float32, shape=[None, None, 1]
        y_pred = self.network() 
        self.loss = tf.reduce_mean(tf.square(y_pred - y), name='loss')  

        return self.loss 


