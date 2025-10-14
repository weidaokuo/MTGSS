import sys
import os
current_file_path = os.path.abspath(__file__)
current_file_path = os.path.dirname(current_file_path)
sys.path.append(current_file_path)
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Args import args
from Moe_Model.main_model import Strain_Stress_Encoder
import itertools
from obtain_HEA_features import Wdk_made_feature
from data.tools.LLM_encode import LLM_process
from data.tools.utils import function
import multiprocessing
from typing import Dict, Any
from itertools import chain
from collections import deque
import pandas as pd





class batch_processing(object):

    def __init__(self, args):
    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Strain_Stress_Encoder(args).to(self.device)   
        self.csv_path = args.csv_path
        self.extract_Paper_Num = args.predict_csv_Num
        self.Alloy_Num_list = args.Alloy_Num_list  
        self.mean_std_array_path = " "               
        self.exepriment_data = self._read_csv_data()        
       
        self.mean_std_array = np.loadtxt(self.mean_std_array_path)
        self.stress_mean = self.mean_std_array[-1, 0]
        self.stress_std = self.mean_std_array[-1, 1]        
        strain_mean = self.mean_std_array[-2, 0]
        strain_std = self.mean_std_array[-2, 1]                
        strain_array = np.arange(0, 100, 0.004)
        self.strain_array_nor = (strain_array-strain_mean)/strain_std               
        self.sqs_length = args.sqs_length
                      
    def get_data(self):                
        curvle_test_tuples = function.csv_predict_data(
                                        Paper_Num=self.extract_Paper_Num, 
                                        Alloy_Num_list = self.Alloy_Num_list,
                                        exepriment_data=self.exepriment_data, 
                                        element_name_list=args.element_name_list, 
                                        physic_feature_list=args.physic_feature_list
                                        )
        
        curvle_tuples_norm = []
        
        for physic_feature, language_data, name in curvle_test_tuples:
            n = physic_feature.shape[1] 
            physic_feature_norm = (physic_feature - self.mean_std_array[:n, 0]) / self.mean_std_array[:n, 1]
            curvle_tuples_norm.append((physic_feature_norm, language_data, name))
        
        return curvle_tuples_norm
        
    def test(self, checkpoint_path): 
        curvle_tuples_norm = self.get_data()
        print("Test start on {}:".format(self.device)) 
        strain_array_segment_list, start_index_list = self.get_strain_input(self.strain_array_nor, num_segments=10)  # εmax=0.2
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval() 
        
        with torch.no_grad():            
            for physic_feature_input, language_input, name in curvle_tuples_norm:
            
                if isinstance(physic_feature_input, torch.Tensor):
                    physic_feature_input = physic_feature_input.cpu().numpy()
                        
                physic_feature_input = np.tile(physic_feature_input, (self.sqs_length, 1))
                physic_feature_input = torch.from_numpy(physic_feature_input).to(dtype=torch.float32).to(self.device)
                physic_feature_input = physic_feature_input.unsqueeze(0)                
                
                language_input = language_input.reshape(1, -1)
                
                if isinstance(language_input, np.ndarray): 
                    language_input = torch.from_numpy(language_input).to(dtype=torch.float32).to(self.device)
                elif isinstance(language_input, torch.Tensor):  
                    language_input = language_input.to(dtype=torch.float32).to(self.device)
                else:
                    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(language_input)}")
                                        
                #language_input = torch.from_numpy(language_input).to(dtype=torch.float32).to(self.device)
                language_input = language_input.unsqueeze(0)                
                
                curvle_list = []
                strain_list = []
                weights_list5 = []
                indices_list5 = []
                for strain, start_index in zip(strain_array_segment_list, start_index_list):
                    strain_input = torch.tensor(strain).to(dtype=torch.float32).to(self.device)
                    strain_input = strain_input.unsqueeze(1).unsqueeze(0)                    
                    weights_list, indices_list, outputs = self.model(physic_feature_input, strain_input, language_input, [start_index])  #[1, 512]                    
                    curvle_list.append(outputs.cpu().numpy())
                    strain_list.append(strain_input.cpu().numpy().reshape(-1, 1))
                    weights5 = weights_list[0].cpu().numpy()
                    indices5 = indices_list[0].cpu().numpy()
                    weights_list5.append(weights5)    
                    indices_list5.append(indices5)                  
                output_weights = np.vstack(weights_list5) 
                output_indices = np.vstack(indices_list5)
                output_strain = np.vstack(strain_list)
                output_moe_data = np.hstack([output_strain, output_weights, output_indices])
                self.process_and_save_data(
                                           curvle_list,                                            
                                           output_file="{}.txt".format(name)
                                           )              
       
    def _read_csv_data(self):    
        csv_data = pd.read_excel(self.csv_path, skiprows=1)
        csv_data = csv_data.iloc[:, 1:]
        exepriment_data = csv_data.loc[csv_data["Paper_Num"].isin(self.extract_Paper_Num)]
        return exepriment_data    
    
    def get_strain_input(self, strain_input, num_segments, segment_length=512):
        segments = []
        start_indices = []
        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = strain_input[start:end]
            segments.append(segment)
            start_indices.append(start)
        return segments, start_indices    
    
    def process_and_save_data(self, data_list, output_file, downsample = False, downsample_factor=10):
        stacked_data = np.hstack(data_list)        
        denormalized_data = stacked_data * self.stress_std + self.stress_mean        
        denormalized_data = denormalized_data.reshape(-1, 1)[:5120]        
        if downsample:
            denormalized_data = denormalized_data[::downsample_factor]
        np.savetxt(output_file, denormalized_data)      
    
    

        
if __name__=='__main__':

    downstream_predict = batch_processing(args)
    downstream_predict.test(checkpoint_path="")


           
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
