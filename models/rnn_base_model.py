# workaround to load data generator module 
# ------------------------------
import sys 
sys.path.insert(0, '/home/andreas_teller/Projects/time_series_data_generator/lib')
# ------------------------------
from collections import deque
from data_generator import DataGeneratorTimeSeries  
from datetime import datetime

import csv
import itertools 
import logging
import numpy as np
import os
import pprint as pp
import tensorflow as tf
import time 

class RNNBaseModel(object):
    """ Interface containing boilerplate code for training tensorflow RNN  
    models. Subclassing models must implement self.calculate_loss(), which 
    returns a  batch loss tensor and self.network() to specifiy the desired
    network structure. 
    Code for the network training, parameter updates, checkpointing, and 
    inference are implemented here and subclasses are mainly responsible 
    for building the computational graph beginning with the placeholders 
    and ending with the loss tensor.
    
    Args:
        train_data_fp (:obj:'str'): Filepath to training data.
        val_data_fp (:obj:'str'): Filepath to validation data.
        hrzn (int): Forecasting horizon which determines the number of time steps 
            that are to be predicted by the model. 
        batch_sizes (:obj:'list' of :obj:'int'): List of mini batch sizes.
        learning_rates (:obj:'list' of :obj:'float'): List of learning rates.
        optimizers (:obj:'list' of :obj:'str'): List of optimization
            algorithms. 'rms' for RMSProp, 'adam' for Adam, 'sgd' for SGD. 
        grad_clips (:obj:'list' of :obj:'int'): List of gradient clip values. 
            Gradients are clipped elementwise to receive norm at most equal to
            clip values.
        early_stopping_steps: Number of steps after which training is stopped
            when no performance increasement was detected.
        min_steps_to_checkpoint: Model only saves after min_steps_to_checkpoint training steps
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
                 train_data_fp, 
                 val_data_fp,
                 hrzn=1, 
                 batch_sizes=[128], 
                 seq_lengths = [10],
                 num_epochs=9999, 
                 learning_rates=[0.1], 
                 optimizers=['adam'], 
                 stp_after=100, 
                 grad_clips=[5],  
                 model_name='RNN_Model', 
                 project_dir=None, 
                 hyper_metrics_fp=project_dir +
                    '/performance_reports/hyper_search.csv':
        
        # Placeholders of hyperparamters for initial graph building. 
        self.batch_size = batch_sizes[0] 
        self.seq_length = seq_lengths[0]   
        self.learning_rate = learning_rates[0] 
        self.grad_clip = grad_clips[0] 
        self.optimizer = optimizers[0] 
        
        # Model name and  filepaths to training data, validation data
        # and where to save the hyperparameter search performance 
        # metrics. 
        self._model_name = model_name 
        self._hyper_metrics_fp = hyper_metrics_fp 
        self._train_data_fp = train_data_fp 
        self._val_data_fp = val_data_fp 
        
        # Hyperparamters. 
        self._seq_lengths = seq_lengths
        self._forec_horiz = hrzn  
        self._optimizers = optimizers
        self._batch_sizes = batch_sizes
        self._learning_rates = learning_rates
        self._num_epochs = num_epochs
        self._grad_clips = grad_clips 

        # Validation metric saving and early stopping variables. 
        self._path = os.path.dirname(__file__) 
        self._save_metrics_fp = self._hyper_metrics_fp 
        self.best_val_metric = None 
        self.stp_counter = 0
        self.stp_after = stp_after   
        self._saver = tf.train.Saver()  

        # TensorFlow graph and session variables.  
        self._graph = self.build_graph() 
        self._session = tf.Session(graph=self._graph) 


    def calculate_loss(self):
        """ The calculation of the loss function has to be implemented by all 
        subclasses.

        Raises: 
            NotImplementedError: If method is not implemented by subclass. 
        """
        raise NotImplementedError('subclass must implement this')

    def network(self):
        """ The network topology has to be implemented by all subclasses. 

        Raise: 
            NotImplementedError: If network typology is not implemented in 
                subclass. 
        """
        raise NotImplementedError('subclass must implement this') 


    def get_optimizer(self, learning_rate):
        """ Returns the specified optimization algorithm. At this moment, 
        Adam, RMSprop, and (classical) gradient descent can be selected.    

        Args: 
            learning_rate (float): The learning rate of an optimizer 
                specifies the magnitude of the movement towards the (local) 
                minimum of the loss function. 

        Returns: 
            (:obj:'tf.optimizer'): TensorFlow optimizer
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


    def __calc_avg_loss(self, loss_hist):
        """ Computes the average of the loss history for the training or 
        validation step per epoch by dividing the sum of all losses by the 
        number of losses.

        Args:
            loss_hist (:obj:'list' of :obj:'float'): The loss history of the
                training or validation step. Each element of the list
                represents the loss of one batch.
        
        Returns:
            float: Average batch loss of epoch.
        """
        return sum(loss_hist) / len(loss_hist)

    def save_metrics(self, last_epoch):
        """ Saves the hyper parameter search validation performance metric
        togehter with the model's name and hyperparamters in one csv file.
        """
        writer = csv.writer(open(self._save_metrics_fp, 'a+'))

        # Check if csv file is empty. If true create header.
        if os.stat(self._save_metrics_fp).st_size == 0 :
            writer.writerow(['model_name', 'seq_length', 'batch_size', 
                             'lear_rate', 'optimizer', 'grad_clip',
                             'last_epoch', 'val_loss']) 
            writer.writerow([self._model_name, self.seq_length,
                             self.batch_size, self.learning_rate, 
                             self.optimizer, self.grad_clip, last_epoch, 
                             self.best_val_metric]) 
        else:     
            writer.writerow([self._model_name, self.seq_length,
                             self.batch_size, self.learning_rate, 
                             self.optimizer, self.grad_clip, last_epoch, 
                             self.best_val_metric]) 


    def report_metrics(self, train_loss_hist, val_loss_hist, ep, train_time):  
        """ Reports the average batch metrics which are printed after each
        epoch or a specified number of epochs. 

        Args: 
            train_loss_hist (:obj:'list' of :obj:'float'): Contains the 
                aggregated loss of each batch in the epoch for the training 
                data.  
            val_loss_hist (:obj:'list' of :obj:'float'): Contains the 
                aggregated loss of each batch in the epoch for the 
                validation data. 
        """
        # compute average losses for training and validation data 
        avg_batch_train_loss = self.__calc_avg_loss(train_loss_hist) 
        avg_batch_val_loss = self.__calc_avg_loss(val_loss_hist) 
        
        # print metrics 
        print(('Epoch: {:4d}, Train Loss: {:12.8f}, Val Loss: {:12.8f}, Time: '
               '{:5.3f} s, Min. Val Loss: {:12.8f}').format(ep+1, 
                    round(avg_batch_train_loss, 8), 
                    round(avg_batch_val_loss, 8),
                    round(train_time, 3), 
                    round(self.best_val_metric, 8)))  

    
    def early_stop(self, val_metric, minimize=True):  
        """ Early stopping aborts the network training process when no 
        validation loss/accuracy improvements were observed for a specified 
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
        if self.best_val_metric is None: 
           self.best_val_metric = val_metric

        # in following epochs  
        else: 
            if val_metric < self.best_val_metric: 
                self.best_val_metric = val_metric
                self.stp_counter = 0 
            else: 
                self.stp_counter += 1 
       
        # check if limit of epochs without any improvement has been 
        # reached 
        if self.stp_counter >= self.stp_after: 
            return False 
        else: 
            return True 


    def train(self, mode='hyp'): 
        """
        Args: 
            mode (:obj:'str', optional): 
        """
        print(80*'-') 
        print(('Initialized Model - Name: {}, Seq. Length: {}, ' 
               'Batch Size: {}, Learning Rate: {}, Optimizer: {}, ' 
               'Grad. Clip.: {}').format(self._model_name, self.seq_length, 
                                         self.batch_size, self.learning_rate, 
                                         self.optimizer, self.grad_clip))   
        
        with self._session.as_default(): 
            
            self._session.run(self.init) 
            ep = 0 
            
            # Initialze the training and validaton data generators which 
            # yield sequence batches for the model training and validation. 
            train_data_gen = DataGeneratorTimeSeries(self.seq_length, 
                                                     self._forec_horiz, 
                                                     self.batch_size, 
                                                     self._train_data_fp)  
            val_data_gen = DataGeneratorTimeSeries(self.seq_length, 
                                                   self._forec_horiz, 
                                                   self.batch_size, 
                                                   self._val_data_fp) 

            while ep < self._num_epochs: 
                
                # Execution of the trainining step.
                train_start = time.time()
                train_loss_hist = []
                
                while train_data_gen.yield_batches():
                    train_batch = train_data_gen.create_batches()
                    train_feed_dict = {getattr(self, placeholder_name, None) : data
                        for placeholder_name, data in train_batch.items() if
                        hasattr(self, placeholder_name)}

                    loss, _ = self._session.run(fetches=[self.loss, self.step],
                        feed_dict = train_feed_dict) 
                    
                    train_loss_hist.append(loss)
                
                # Stop the training timer and compute the time it took to train 
                # one epoch. 
                train_end = time.time()
                train_time = train_end - train_start

                # Execution of the validation step in which no paramaters
                # updates are performed and only the loss for the
                # validation data is computed.  
                val_loss_hist = []
                while val_data_gen.yield_batches():
                    val_batch = val_data_gen.create_batches()
                    val_feed_dict = {getattr(self, placeholder_name, None) : data
                        for placeholder_name, data in val_batch.items() if
                        hasattr(self, placeholder_name)}
                    loss = self._session.run(fetches=[self.loss],
                        feed_dict = val_feed_dict)
                    val_loss_hist.append(loss[0])
                
                # Early stopping stops the training when a specified number
                # of epochs without any improvement in the validation metric
                # has passed.

                val_metric = self.__calc_avg_loss(val_loss_hist)  
                if not (self.early_stop(val_metric)):
                    if mode == 'hyp': 
                        self.save_metrics(ep) 
                    if mode == 'train': 
                        self.save_model(self._saver, val_metric,
                                        self._session, self._model_name)   
                    
                    return

                # Reset the training  validation data generator for the next 
                # epoch and report metrics.  
                train_data_gen.reset_gen()
                val_data_gen.reset_gen()
                self.report_metrics(train_loss_hist, val_loss_hist, ep, train_time)
                ep += 1
        
        # When the training is run in hyperparamter search mode only save 
        # the validation metrics and do not save models. 
        if mode == 'hyp': 
            self.save_metrics(ep)

        if mode == 'train': 
        self.save_model(self._saver, val_metric, self._session,
                        self._model_name)   
              

    def hyp_search(self): 
        """ Iteratively computes the Cartesian product of all hyperparameters
        and invokes the training process for each element.
        """
        for i in itertools.product(self._batch_sizes, 
                                   self._learning_rates,
                                   self._seq_lengths, 
                                   self._grad_clips,
                                   self._optimizers): 
            self.batch_size = i[0] 
            self.learning_rate = i[1] 
            self.seq_length = i[2] 
            self.grad_clip = i[3] 
            self.optimizer = i[4] 

            # Updates the tensorflow computational graph with
            # the new hyperparameters. 
            self._graph = self.build_graph() 
            self._session = tf.Session(graph=self._graph) 

            # Run network training for the specified 
            # hyperparameter combinations in hyp mode
            # which means that models are not saved. 
            self.train(mode='hyp') 

            # Reset the saved best validation metric before new
            # hyperparameter combination is tested.  
            self.best_val_metric = None  

    def fit(self): 
        self.train(mode='train') 

    def prediction(self): 
        # TODO: implement prediction step 
        pass

    # TODO: implement learning rate reductions 

    def save_model(self, saver, validation_metric, session, name):
        """
        Args: 
            saver () 
            validation_metric (): 
            sesseon (): 
            name (): 
        """
        if self.best_val_metric == None: 
            if not os.path.isdir('saved_models_' + name):
                os.mkdir('saved_models_' + name)             
        else:  
            if validation_metric < self.best_val_metric:         
                save_path = saver.save(session, "./saved_models_" + name + "/model.ckpt")
                print("Model saved")

    def build_graph(self): 
        """
        """
        with tf.Graph().as_default() as graph: 
            self.global_step = tf.Variable(0, trainable=False) 
            self.loss = self.calculate_loss() 
            self.update_parameters(self.loss) 
            self.init = tf.global_variables_initializer()

            return graph


    def update_parameters(self, loss): 
        """
        Args: 
            loss (float): 
        """
        # Chosen optimizer is received and gradiens are computed and clipped. 
        optimizer = self.get_optimizer(self.learning_rate) 
        grads = optimizer.compute_gradients(loss) 
        clipped = [(tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v_)
                   for g, v_ in grads] 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
        
        with tf.control_dependencies(update_ops): 
            step = optimizer.apply_gradients(clipped, 
                                             global_step=self.global_step) 

        self.step = step 
