# workaround to load the data generator module 
import sys 
sys.path.insert(0, '/home/andreas_teller/Projects/time_series_data_generator/lib')

# standard imports 
from data_generator import DataGeneratorTimeSeries  
from tom_sawyer_toolbox.models.lstm import LSTM

if __name__ == '__main__': 
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    test_data_fp = '/home/andreas_teller/Projects/tom_sawyer_toolbox/test_data/processed/sinusoid.csv' 

    
    data_gen = DataGeneratorTimeSeries(10, 1, 64, test_data_fp) 
    model = LSTM(data_gen=data_gen,  
                 num_time_steps=10, 
                 learning_rate=0.01,  
                 batch_size=64,  
                 optimizer='adam', 
                 num_epochs=1000, 
                 grad_clip=5, 
                 num_units=256) 

    model.train() 
