import os
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import os
import random
import pickle
import lmdb
from transformers import AutoTokenizer, AutoModel
from typing import Literal
#from data_trans_lmdb import data_to_lmdb





class Model_Select(object):

    @classmethod
    def steelbert(cls, text_input, device):
        """
        Use the SteelBERT model to encode the process.
        texts = ["A composite steel plate for marine construction was fabricated using 316L stainless steel."]       
        """
        #print("text_input:", text_input) 
        # Load the locally downloaded SteelBERT model
        model_path = parent_dir + "/llm/steelbert"    
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)        # Move model to GPU if available
        inputs = tokenizer(text_input, return_tensors='pt', padding=True, truncation=False).to(device)         
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        # Hidden states of all layers 
        hidden_states = outputs.hidden_states   
        
        # Hidden states of the last layer
        last_hidden_state = hidden_states[-1]  
        
        # Retrieve the vector corresponding to the CLS token. the shape of cls_embeddings:[1, 768]
        cls_embeddings = last_hidden_state[:, 0, :]                  
        #print("cls_embeddings.shape:",cls_embeddings.shape)
        
        return cls_embeddings
        
    @classmethod
    def qwen(cls, text_input, device):
        print("the qwen model is not finished! please use other model.")
        pass



class LLM_process(Model_Select):
    """
    Processing flow：
        1、self.csv_path is the path to the CSV file containing the composition, processing, and physical features.

        2、self._read_data() completes the conversion of textual descriptions of composition and process and saves them to self.data.
    """

    def __init__(self, model_name = Literal["steelbert", "qwen"], device=True):
        super().__init__()
    
        """
        :param composition_element_list is a list like ["Mo", "Nb", "Ta"]
        :param composition_fraction_list is a list like [0.96, 0.2, 0.5].（weight fraction）
        """
        self.model_name = model_name
        if device:            
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
          
    def composition_encode(self, element_list:list, fraction_list:list):
        value_list = []
        if hasattr(self, self.model_name):
            for i, j in zip(element_list, fraction_list):
                ouput = getattr(self, self.model_name)([i], self.device)*j
                value_list.append(ouput)
        stacked_tensors = torch.stack(value_list, dim=0)
        summed_tensor = torch.sum(stacked_tensors, dim=0)
        return summed_tensor
    
    def direct_encode(self, process_list:list):
        if hasattr(self, self.model_name):
            ouput = getattr(self, self.model_name)(process_list, self.device)
        return ouput
        
    def steelbert_method(self, element_list:list, fraction_list:list, process_list:list): 
        """
        The shape of steelbert_output is [1, 1536].
        """
        fraction_list = LLM_process.fraction_list_charge(fraction_list)
        tensor1 = self.composition_encode(element_list, fraction_list)
        tensor2 = self.direct_encode(process_list)
        steelbert_output = torch.cat((tensor1, tensor2), dim=1)
        return steelbert_output
    
    @staticmethod
    def fraction_list_charge(lst):
        """
        Determine the maximum value in the list:
        - If the maximum value is <= 1, no changes are made.
        - If there are values greater than 1, divide all values by 100.
        """
        if max(lst) > 1:
            return [x / 100 for x in lst]
        return lst    


if __name__=='__main__':
    model = LLM_process("steelbert")
    output = model.composition_encode(["Nb", "Hf", "C"], [96, 3, 1])
    output1 = model.direct_encode(["we are using rolling and melt method as it is processing method."])
    steelbert_output = model.steelbert_method(["Nb", "Hf", "C"], [96, 3, 1], ["we are using rolling and melt method as it is processing method."])
    steelbert_output = steelbert_output.cpu().numpy()
    print(steelbert_output, steelbert_output.shape)
























    
        
        
        
        
        
        
        
        