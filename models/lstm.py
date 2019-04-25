import tensorflow as tf 


class LSTM(base_rnn_model): 
    def __init__(self, train_data, val_data, batch_size, lr, look_back,
            dropout, units, epochs, **kwargs): 
        self._train_data = train_data 
        self._val_data = val_data 

    def calculate_loss(self, y_pred, y_true): 
        return tf.reduce_mean(tf.square(y_pred - y), name='loss')  

    def network(self): 

    def train(self): 

        # reset tf graph 
        tf.reset_default_graph() 

        x_train = tf.placeholder(tf.float32, shape=[self._batch_size,
            self._num_time_steps, self._dim], name='x_train') 

        y_train = tf.placheholder(tf.float32, shape=[self._batch_size,
            self._num_time_steps, self._dim], name='y_train') 

        # TODO implement network

        # define loss function 
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

            

