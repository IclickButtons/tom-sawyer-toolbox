from collections import deque
from datetime import datetime
import time 
import logging
import os
import pprint as pp
import numpy as np
import tensorflow as tf

class RNNBaseModel(object):

    """Interface containing boilerplate code for training tensorflow RNN  
    models. Subclassing models must implement self.calculate_loss(), which 
    returns a tensor for the batch loss. Code for the training loop, 
    parameter updates, checkpointing, and inference are implemented here and
    subclasses are mainly responsible for building the computational graph 
    beginning with the placeholders and ending with the loss tensor.
    Args:
        data_gen: Data generator class that yields batch dictionaries mapping 
            tf.placeholder names (as strings) to batch data (numpy arrays).
        batch_size: Minibatch size.
        learning_rate: Learning rate.
        optimizer: 'rms' for RMSProp, 'adam' for Adam, 'sgd' for SGD
        grad_clip: Clip gradients elementwise to have norm at most equal to grad_clip.
        regularization_constant:  Regularization constant applied to all trainable parameters.
        keep_prob: 1 - p, where p is the dropout probability
        early_stopping_steps:  Number of steps to continue training after validation loss has
            stopped decreasing.
        warm_start_init_step:  If nonzero, model will resume training a restored model beginning
            at warm_start_init_step.
        num_restarts:  After validation loss plateaus, the best checkpoint will be restored and the
            learning rate will be halved.  This process will repeat num_restarts times.
        enable_parameter_averaging:  If true, model saves exponential weighted averages of parameters
            to separate checkpoint file.
        min_steps_to_checkpoint:  Model only saves after min_steps_to_checkpoint training steps
            have passed.
        log_interval:  Train and validation accuracies are logged every log_interval training steps.
        loss_averaging_window:  Train/validation losses are averaged over the last loss_averaging_window
            training steps.
        num_validation_batches:  Number of batches to be used in validation evaluation at each step.
        log_dir: Directory where logs are written.
        checkpoint_dir: Directory where checkpoints are saved.
        prediction_dir: Directory where predictions/outputs are saved.
    """

    def __init__(self,
                 data_gen, 
                 batch_size=128, 
                 num_epochs= 1000, 
                 learning_rate=0.1, 
                 optimizer='adam', 
                 stp_after=100, 
                 grad_clip=5):

        self._batch_size = batch_size
        self._num_epochs = num_epochs

        self._data_gen = data_gen

        self.optimizer = optimizer

        self.best_validation_metric = None 
        self.stp_counter = 0
        self.stp_after = stp_after   
        self._grad_clip = grad_clip 

        self._graph = self.build_graph() 
        self._session = tf.Session(graph=self._graph) 
    
    def calculate_loss(self):
        """ The calculation of the loss function has to be implemented by all 
        subclasses. Necessarily, this also includes the specification of the
        specfic model typoology. 

        Raises: 
            NotImplementedError: If method is not implemented by subclass. 
        """
        raise NotImplementedError('subclass must implement this')


    def get_optimizer(self, learning_rate):
        """ Returns the specified optimization algorithm. At this moment, 
        Adam, RMSprop, and (classical) gradient descant can be chosen.    

        Args: 
            learning_rate (float): The learning rate of an optimizer 
                specifies the magnitude of the movement towards the (local) 
                minimum of the loss function. 

        Returns: Tensorflow optimizer  
        """
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate, 
                                          name='optimizer')
        elif self.optimizer == 'gd':
            return tf.train.GradientDescentOptimizer(learning_rate, 
                                                     name='optimizer')
        elif self.optimizer == 'rms':
            return tf.train.RMSPropOptimizer(learning_rate, 
                                             decay=0.95, 
                                             momentum=0.9, 
                                             name='optimizer')
        else:
            assert False, 'optimizer must be adam, gd, or rms'


    def report_metrics(self, train_loss_hist, ep, train_time): # val_loss_hist, ep): 
        """ Reports the average batch metrics which are printed after each
        epoch or a specified number of epochs. 

        Args: 
            train_loss_hist (list of floats): contains the aggregated loss of
                each batch in the epoch for the training data.  
            val_loss_hist (list of floats): contains the aggregated loss of
                each batch in the epoch for the validation data. 
        """
        # compute average losses for training and validation data 
        avg_batch_train_loss= sum(train_loss_hist) / len(train_loss_hist) 
        #avg_batch_val_loss = sum(val_loss_hist) / len(val_loss_hist) 
        
        # print metrics 
        print('Epoch: {}, Training Loss: {}, Validation Loss: {}, Time: {} s'.format(ep, 
            avg_batch_train_loss, 1, round(train_time, 3))) # avg_batch_val_loss))  

    
    def early_stopping(self, val_metric, minimize=True):  
        """ Early stopping aborts the network training process when no 
        validation loss/accuracy improvements were observed for a speciefied 
        amount of epochs defined in the variable stp_after. 

        Args: 
            val_metric (float): The metric which is validated after
                each epoch, e.g., the accuracy or loss.   
            minimize (boolean, optional): Specifies if the validation 
                metric should be minimized (loss)  or maximized (accuracy). 
                Defaults to True.

        Returns: 
            bool: True, if training process should be continued, False 
                otherwise. 
        """
        # behaviour in first epoch 
        if self.best_validation_metric is None: 
           self.best_validation_metric = val_metric

        # in following epochs  
        else: 
            if val_metric < self.best_validation_metric: 
                self.best_validation_metric = val_metric
                self.stp_counter = 0 
            else: 
                self.stp_counter += 1 
       
        # check if limit of epochs without any improvement has been 
        # reached 
        if self.stp_counter >= self.stp_after: 
            return False 
        else: 
            return True 

    def train(self): 
        with self._session.as_default(): 
            
            self._session.run(self.init) 
            ep = 0 
             
            while ep < self._num_epochs: 
                
                # trainining step  
                train_start = time.time()
                ep_loss_hist = []  
                
                while self._data_gen.yield_batches(): 
                    train_batch = self._data_gen.create_batches() 

                    train_feed_dict = {getattr(self, placeholder_name, None) : data
                        for placeholder_name, data in train_batch.items() if
                        hasattr(self, placeholder_name)}

                    loss, _ = self._session.run(fetches=[self.loss, self.step],
                        feed_dict = train_feed_dict) 
                    
                    ep_loss_hist.append(loss) 
                
                train_end = time.time() 
                train_time = train_end - train_start 
                self.report_metrics(ep_loss_hist, ep, train_time)

                ep += 1

                # Reset the data generator for the next epoch. 
                self._data_gen.reset_gen()

                    # TODO implement validation step 
                    
                    # TODO implement metric reports 

    def save_model(self, saver, validation_metric, session, name):
        if self.best_validation_metric == None: 
            if not os.path.isdir('saved_models_' + name):
                os.mkdir('saved_models_' + name)             
        else:  
            if validation_metric < self.best_validation_metric:         
                save_path = saver.save(session, "./saved_models_" + name + "/model.ckpt")
                print("Model saved")

    def build_graph(self): 
        with tf.Graph().as_default() as graph: 
            self.global_step = tf.Variable(0, trainable=False) 
            self.loss = self.calculate_loss() 
            self.update_parameters(self.loss) 
            self.init = tf.global_variables_initializer()

            return graph


    def update_parameters(self, loss): 
        optimizer = self.get_optimizer(0.001) 
        grads = optimizer.compute_gradients(loss) 
        clipped = [(tf.clip_by_value(g, -self._grad_clip, self._grad_clip), v_)
            for g, v_ in grads] 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
        with tf.control_dependencies(update_ops): 
            step = optimizer.apply_gradients(clipped,
                global_step=self.global_step) 

        self.step = step 
