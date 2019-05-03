# TODO: implement  

if __name__ == '__main__': 
    data_train_fp = '/home/andreas_teller/Projects/tom_sawyer_toolbox/test_data/processed/train/sinusoid.csv' 
    data_val_fp = '/home/andreas_teller/Projects/tom_sawyer_toolbox/test_data/processed/val/sinusoid.csv' 
    data_test_fp = '' 

    model = LSTM(train_data_fp=data_train_fp,  
                 val_data_fp=data_val_fp,  
                 seq_lengths=[10], 
                 num_pred=1, 
                 learning_rates=[0.01],  
                 batch_sizes=[64],   
                 optimizers=['adam'], 
                 num_epochs=2, 
                 grad_clips=[5], 
                 num_units=256, 
                 model_name='2-Layer-LSTM') 

    model.fit()  
     
