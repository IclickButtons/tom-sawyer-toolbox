import tensorflow as tf 

class LSTM(base_rnn_model): 
    def __init__(self, train_data, val_data, batch_size, lr, look_back,
            dropout, units, epochs, **kwargs): 
        self._train_data = train_data 
        self._val_data = val_data 

    def network(self)
    # TODO implement network
    layer1 = 
    layer2 = 
    layer3 = 

    def calculate_loss(self, y_pred, y_true): 
        self.x_train = tf.placeholder(tf.float32, shape=[None, None, self._D] 
        self.y_train = tf.placeholder(tf.float32, shape=[None, None, 1]
        return tf.reduce_mean(tf.square(y_pred - y), name='loss')  


    def train(self): 

        # reset tf graph 
        tf.reset_default_graph() 



        # TODO add logger 
        loss = self.calculate_loss(y_pred, y_true) 

        # define optimizer 
        with tf.name_scope('optimizer'): 
            optimizer = self.get_optimizer(self._lr)

        # initialize variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as session: 
            try: 
                session.run(init_op) 
                # Logger Variables have been initialized

                # iterate over epochs 
                for ep in range(self._num_epochs): 
                    # trainining 
                    train_loss_hist = # feed step 

                    # validation 

                    val_loss_hist = # feed step 

            
if __name__ == '__main__': 
    data_gen = TimeSeriesDataGenerator() 
    lstm = LSTM() 
