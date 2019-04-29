# workaround for now to load the data generator module 
import sys 
#sys.path.insert(0, '/home/andreas/github_repos/time-series-data-generator/lib')

# standard imports 
#from data_generator import DataGeneratorTimeSeries  
from tom_sawyer_toolbox.models.lstm import LSTM

if __name__ == '__main__': 
    data_gen = DataGeneratorTimeSeries(data_dir='data/processed') 
    model = LSTM(data_gen=data_gen,  
                 learning_rates=[0.01, 0.001],  
                 batch_sizes=[64, 128, 256],  
                 optimizer='adam', 
                 num_epochs=1000, 
                 grad_clip=5, 
                 num_units=256, 
                 num_time_steps=10)

    model.train() 
