import lmdb
import numpy as np
import pickle
import torch
import pandas as pd



class data_to_lmdb(object):


    def __init__(self, lmdb_path, map_size=2**41):
    
        self.lmdb_path = lmdb_path
        self.map_size = map_size
        
    def save_to_lmdb(self, tuples):  # 1TB
        """ Save tuples of (data_input, data_output) to LMDB """
        env = lmdb.open(self.lmdb_path, map_size=int(self.map_size))
        
        with env.begin(write=True) as txn:
            for i, (physic_feature_input, strain_input, language_input, stress_output, start_index) in enumerate(tuples):
                # Convert data to bytes
                input_bytes = self.serialize_data(physic_feature_input)
                output_bytes = self.serialize_data(strain_input)
                input_temp_bytes = self.serialize_data(language_input)
                stress_output = self.serialize_data(stress_output)
                start_index = self.serialize_data(start_index)
                
                
                # Combine into a single dictionary for LMDB storage
                data_dict = {'physic_feature_input': input_bytes, 'strain_input': output_bytes, "language_input":input_temp_bytes, "stress_output":stress_output, "start_index":start_index}
                
                # Serialize the dictionary and store it
                data_bytes = self.serialize_data(data_dict)
                
                # Store the data in LMDB
                txn.put(f'{i}'.encode(), data_bytes)
        
        env.close()
    
    
    def serialize_data(self, data):
        """ Serialize data into bytes """
        return pickle.dumps(data)
    

        



class curvle_data_to_lmdb(object):


    def __init__(self, lmdb_path, map_size=2**41):
    
        self.lmdb_path = lmdb_path
        self.map_size = map_size
        
    def save_to_lmdb(self, tuples):  # 1TB
        """ Save tuples of (data_input, data_output) to LMDB """
        env = lmdb.open(self.lmdb_path, map_size=int(self.map_size))
        
        with env.begin(write=True) as txn:
            for i, (physic_feature_input, language_input, strain_stress_curvle) in enumerate(tuples):
                # Convert data to bytes
                input_bytes = self.serialize_data(physic_feature_input)
                input_temp_bytes = self.serialize_data(language_input)
                stress_output = self.serialize_data(strain_stress_curvle)
                
                
                # Combine into a single dictionary for LMDB storage
                data_dict = {'physic_feature_input': input_bytes,  "language_input":input_temp_bytes, "stress_output":stress_output}
                
                # Serialize the dictionary and store it
                data_bytes = self.serialize_data(data_dict)
                
                # Store the data in LMDB
                txn.put(f'{i}'.encode(), data_bytes)
        
        env.close()
    
    
    def serialize_data(self, data):
        """ Serialize data into bytes """
        return pickle.dumps(data)




class downstream_to_lmdb(object):


    def __init__(self, lmdb_path, map_size=2**41):
    
        self.lmdb_path = lmdb_path
        self.map_size = map_size
        
    def save_to_lmdb(self, tuples):  # 1TB
        """ Save tuples of (data_input, data_output) to LMDB """
        env = lmdb.open(self.lmdb_path, map_size=int(self.map_size))
        
        with env.begin(write=True) as txn:
            for i, (name, physic_feature_input, language_input) in enumerate(tuples):
                # Convert data to bytes
                name = self.serialize_data(name)
                input_bytes = self.serialize_data(physic_feature_input)
                input_temp_bytes = self.serialize_data(language_input)
                
                
                
                # Combine into a single dictionary for LMDB storage
                data_dict = {'name':name, 'physic_feature_input': input_bytes,  "language_input":input_temp_bytes}
                
                # Serialize the dictionary and store it
                data_bytes = self.serialize_data(data_dict)
                
                # Store the data in LMDB
                txn.put(f'{i}'.encode(), data_bytes)
        
        env.close()
    
    
    def serialize_data(self, data):
        """ Serialize data into bytes """
        return pickle.dumps(data)
    
