import lmdb
import numpy as np
import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class LMBD_Segment_Dataset(Dataset):

    def __init__(self, lmdb_path, sqs_length):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = int(txn.stat()['entries'])  # Get the total number of entries in the database
        self.sqs_length = sqs_length

    def __getitem__(self, index):
    
        with self.env.begin() as txn:
            data_bytes = txn.get(f'{index}'.encode())
            data_dict = self.deserialize_data(data_bytes)
            #print("data_dict", data_dict)
            
            
            physic_feature_input = self.deserialize_data(data_dict['physic_feature_input'])
            strain_input = self.deserialize_data(data_dict['strain_input'])
            language_input = self.deserialize_data(data_dict['language_input'])
            stress_output = self.deserialize_data(data_dict['stress_output'])
            start_index = self.deserialize_data(data_dict['start_index'])
            
            physic_feature_input = np.tile(physic_feature_input, (self.sqs_length, 1))
            strain_input = strain_input.reshape(-1, 1)
            language_input = language_input.reshape(1, -1)
          
            # Convert NumPy arrays to PyTorch Tensors
            physic_feature_input = torch.from_numpy(physic_feature_input).to(dtype=torch.float32)
            strain_input = torch.from_numpy(strain_input).to(dtype=torch.float32)
            language_input = torch.from_numpy(language_input).to(dtype=torch.float32)
            stress_output = torch.from_numpy(stress_output).to(dtype=torch.float32)
            #print("physic_feature_input:", physic_feature_input.shape)
            #print("strain_input:", strain_input.shape)
            #print("language_input:", language_input.shape)
            #print("stress_output:", stress_output.shape)
            #print("start_index:", start_index)
            return physic_feature_input, strain_input, language_input, stress_output, start_index
                        
    def __len__(self):
        return self.length
    
    
    def deserialize_data(self, data_bytes):
        """ Deserialize data from bytes """
        return pickle.loads(data_bytes)            




class LMBD_Curvle_Dataset(Dataset):

    def __init__(self, lmdb_path, sqs_length):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = int(txn.stat()['entries'])  # Get the total number of entries in the database
        self.sqs_length = sqs_length

    def __getitem__(self, index):
    
        with self.env.begin() as txn:
            data_bytes = txn.get(f'{index}'.encode())
            data_dict = self.deserialize_data(data_bytes)
            #print("data_dict", data_dict)
            physic_feature_input = self.deserialize_data(data_dict['physic_feature_input'])
            language_input = self.deserialize_data(data_dict['language_input'])
            stress_output = self.deserialize_data(data_dict['stress_output'])
            valid_length = stress_output.shape[0]
            stress_output = self.padding(stress_output, 20000)
             
            physic_feature_input = np.tile(physic_feature_input, (self.sqs_length, 1))
            language_input = language_input.reshape(1, -1)
             
            # Convert NumPy arrays to PyTorch Tensors
            physic_feature_input = torch.from_numpy(physic_feature_input).to(dtype=torch.float32)
            language_input = torch.from_numpy(language_input).to(dtype=torch.float32)
            stress_output = torch.from_numpy(stress_output).to(dtype=torch.float32)
            #print("stress_output的形状是：", stress_output.shape)
            return physic_feature_input, language_input, stress_output, valid_length
            
    def __len__(self):
        return self.length
    
    def deserialize_data(self, data_bytes):
        """ Deserialize data from bytes """
        return pickle.loads(data_bytes)
    
    def padding(self, curvle_input, length):
        """
        将形状为 [n, 2] 的 numpy 矩阵填充 0 至 [length, 2] 形状。
        参数:
            curvle_input (np.ndarray): 形状为 [n, 2] 的 numpy 矩阵。
            length (int): 目标长度。
        返回:
            np.ndarray: 填充后的形状为 [length, 2] 的 numpy 矩阵。
        """
        n = curvle_input.shape  [0]  # 输入矩阵的长度
        if n >= length:
            # 如果输入矩阵的长度已经大于或等于目标长度，直接返回原矩阵
            raise Exception("曲线的有效长度大于或等于了目标长度，建议增大代码中padding函数的length值")
        else:
            # 计算需要填充的长度
            pad_length = length - n
            # 在末尾填充 0
            padded_matrix = np.pad(curvle_input, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
            return padded_matrix
        

class LMBD_Downstream_Dataset(Dataset):

    def __init__(self, lmdb_path, sqs_length):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = int(txn.stat()['entries'])  # Get the total number of entries in the database
        self.sqs_length = sqs_length

    def __getitem__(self, index):
    
        with self.env.begin() as txn:
            data_bytes = txn.get(f'{index}'.encode())
            data_dict = self.deserialize_data(data_bytes)
            #print("data_dict", data_dict)
            name = self.deserialize_data(data_dict['name'])
            physic_feature_input = self.deserialize_data(data_dict['physic_feature_input'])
            language_input = self.deserialize_data(data_dict['language_input'])
            

             
            physic_feature_input = np.tile(physic_feature_input, (self.sqs_length, 1))
            language_input = language_input.reshape(1, -1)
             
            # Convert NumPy arrays to PyTorch Tensors
            physic_feature_input = torch.from_numpy(physic_feature_input).to(dtype=torch.float32)
            language_input = torch.from_numpy(language_input).to(dtype=torch.float32)
            
            #print("stress_output的形状是：", stress_output.shape)
            return name, physic_feature_input, language_input
            
    def __len__(self):
        return self.length
    
    def deserialize_data(self, data_bytes):
        """ Deserialize data from bytes """
        return pickle.loads(data_bytes)
    

    
if __name__ == '__main__':

    """ 
    lmdb_load_path = "/home/weidaokuo/project/strain_stress/Nb_alloy/data/lmdb_data/train/segment_train"
    # Example usage
    lmdb_dataset = LMBD_Segment_Dataset(lmdb_load_path, 512)
   
    data_loader = DataLoader(lmdb_dataset, batch_size=8, shuffle=True)
    print("Dataset length:", len(lmdb_dataset))  # 检查数据集长度
    # Iterate over the data loader

    for physic_feature_input, strain_input, language_input, stress_output, start_index in data_loader:
        print("physic_feature_input:", physic_feature_input.shape)   #[batch_size, 512, 15]
        print("strain_input:", strain_input.shape)        #[batch_size, 512, 1]
        print("language_input:", language_input.shape)    #[batch_size, 1, 1536]
        print("stress_output:", stress_output.shape)      #[8, 512]
        print("start_index:", start_index)              
    print("Dataset length:", len(lmdb_dataset))  # 检查数据集长度
    """
    
    #用于强化学习训练的数据
    curvle_load_path = "/home/weidaokuo/project/strain_stress/Nb_alloy/data/lmdb_data/test/curvle_test"
    lmdb_curvle_dataset = LMBD_Curvle_Dataset(curvle_load_path, 512) 
    data_loader_curvle = DataLoader(lmdb_curvle_dataset, batch_size=2, shuffle=False)
    print("Dataset length:", len(data_loader_curvle))  # 检查数据集长度
    # Iterate over the data loader

    for physic_feature_input, language_input, stress_output, valid_length in data_loader_curvle:
        print("physic_feature_input:", physic_feature_input.shape)
        print("language_input:", language_input.shape)    #[batch_size, 1, 1536]
        print("stress_output:", stress_output.shape)      #[]
        print("valid_length:", valid_length)

        




















































        
        
        