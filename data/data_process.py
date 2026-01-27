import sys
import os

# Get the absolute path of the current file.
current_file_path = os.path.abspath(__file__)
nb_alloy_path = os.path.dirname(os.path.dirname(current_file_path))

# Add the path to the home directory to sys.path.
if nb_alloy_path not in sys.path:
    sys.path.insert(0, nb_alloy_path)
print(nb_alloy_path)

import numpy as np
import torch
import pandas as pd
import lmdb
from data.tools.data_trans_lmdb import data_to_lmdb
from Args import args
from data.tools.utils import function




class strain_stress_data_process(object):
    """   
    Preprocess the data and save the training set in LMDB format.
    """

    def __init__(self, args):
    
        self.train_all_curvle_path = args.all_curvle_path    
        self.csv_path = args.csv_path
        self.extract_Paper_Num = args.extract_Paper_Num       
        self.all_segment_train_data = data_to_lmdb(args.all_train_segment_lmdb_path)
        self.exepriment_data = self._read_csv_data()    
        
    def get_data(self, normalize_save=True):
        """
        Args:
            normalize_save (bool): Whether to save the mean and standard deviation of the features and stress–strain curves locally.
        """
        # Import of stress–strain curves
        curvle_train_tuples = function.obtain_data(curvle_path=self.train_all_curvle_path, exepriment_data=self.exepriment_data, element_name_list=args.element_name_list, physic_feature_list=args.physic_feature_list)
        # Gaussian noise is added to the training set for data augmentation.      
        curvle_train_tuples = function.add_gaussian_noise(curvle_train_tuples, radio=0.01, n=5)        
        mean_std_array=function.normalize(curvle_train_tuples, normalize_save=True)       
        # Normalize curvle_train_tuples and curvle_test_tuples.
        curvle_train_tuples = function.input_data_norm(curvle_train_tuples, mean_std_array)        
        # save data
        segment_train_tuples = function.trans_segment(curvle_train_tuples, args.split_length, args.overlap_length)
        self.all_segment_train_data.save_to_lmdb(segment_train_tuples)
       
    def _read_csv_data(self):
        csv_data = pd.read_excel(self.csv_path, skiprows=1)
        csv_data = csv_data.iloc[:, 1:]
        exepriment_data = csv_data.loc[csv_data["Paper_Num"].isin(self.extract_Paper_Num)]
        return exepriment_data
        
        
if __name__=='__main__':

    data_process = strain_stress_data_process(args)
    data_process.get_data()



  
        

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
