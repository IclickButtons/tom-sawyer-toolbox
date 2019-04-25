from collections import deque
from datetime import datetime
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
        reader: Class with attributes train_batch_generator, val_batch_generator, and test_batch_generator
            that yield dictionaries mapping tf.placeholder names (as strings) to batch data (numpy arrays).
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

    def __init__(self, optimizer='adam', mode='test', stp_after=100):
        self.optimizer = optimizer
        self.mode = mode
        self.best_validation_metric = None 
        self.stp_counter = 0
        self.stp_after = stp_after   

    
    def calculate_loss(self):
        """ The calculation of the loss function has to be implemented by all 
        subclasses. 

        Raises: 
            NotImplementedError: If method is not implemented by subclass. 
        """
        raise NotImplementedError('subclass must implement this')


    def get_optimizer(self, learning_rate):
        """ Get method which returns the specified optimization algorithm.  

        Args: 
            learning_rate (float): The learning rate of an optimizer 
                specifies the magnitude of the movement towards the (local) 
                minimum of the loss function. 

        Returns: Tensorflow optimizer  
        """
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate, name ='optimizer')
        elif self.optimizer == 'gd':
            return tf.train.GradientDescentOptimizer(learning_rate, name ='optimizer')
        elif self.optimizer == 'rms':
            return tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9, name ='optimizer')
        else:
            assert False, 'optimizer must be adam, gd, or rms'

    def report_metrics(self, train_loss_hist, val_loss_hist, ep): 

        avg_batch_train_loss= sum(train_loss_hist) / len(train_loss_hist) 
        avg_batch_val_loss = sum(val_loss_hist) / len/val_loss_hist) 
        print('Epoch:')  

    
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


    def feed_batch(self, data_generator, mode, session, optimizer, loss, tf_x_placeholder, tf_y_placeholder, prob, dropout): 
	
        loss_hist = []

        while(data_generator.is_full()): 
            u_data, u_labels = data_generator.unroll_batches()
            feed_dict = {}
            feed_dict[tf_x_placeholder] = u_data
            feed_dict[tf_y_placeholder] = u_labels
            
            if mode == 'train':
                feed_dict[prob] = dropout   
                _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                loss_hist.append(l)

            if mode == 'val': 
                l = session.run([loss], feed_dict=feed_dict)
                loss_hist.append(l[0])
        
        return loss_hist

    def feed_batch_pred(self, data_generator, mode, session, tf_x_placeholder, tf_y_placeholder, y_pred): 
	
        pred = []

        while(data_generator.is_full()): 
            u_data, u_labels = data_generator.unroll_batches()
            feed_dict = {}
            feed_dict[tf_x_placeholder] = u_data
            feed_dict[tf_y_placeholder] = u_labels

            if mode == 'predict': 
                l = session.run(y_pred, feed_dict=feed_dict)
                pred.append(l)

        return pred


    def save_model(self, saver, validation_metric, session, name):
        if self.best_validation_metric == None: 
            if not os.path.isdir('saved_models_' + name):
                os.mkdir('saved_models_' + name)             
        else:  
            if validation_metric < self.best_validation_metric:         
                save_path = saver.save(session, "./saved_models_" + name + "/model.ckpt")
                print("Model saved")

