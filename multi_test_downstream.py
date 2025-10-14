import sys
import os
current_file_path = os.path.abspath(__file__)
current_file_path = os.path.dirname(current_file_path)
sys.path.append(current_file_path)
from data.load_dataset import LMBD_Downstream_Dataset
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from Args import args
from Moe_Model.main_model import Strain_Stress_Encoder
import itertools
from obtain_HEA_features import Wdk_made_feature
from data.tools.LLM_encode import LLM_process
from data.tools.data_trans_lmdb import downstream_to_lmdb
import multiprocessing
from typing import Dict, Any
from itertools import chain
from collections import deque



atomic_density = {'H': 8.988e-05, 'He': 0.0001785, 'Li': 0.534, 'Be': 1.85, 'B': 2.34, 'C': 2.267, 'N': 0.0012506,
                  'O': 0.001429, 'F': 0.001696, 'Ne': 0.0008999, 'Na': 0.971, 'Mg': 1.738, 'Al': 2.698,
                  'Si': 2.3296,
                  'P': 1.82, 'S': 2.067, 'Cl': 0.003214, 'Ar': 0.0017837, 'K': 0.862, 'Ca': 1.54, 'Sc': 2.989,
                  'Ti': 4.54, 'V': 6.11, 'Cr': 7.15, 'Mn': 7.21, 'Fe': 7.874, 'Co': 8.86, 'Ni': 8.912, 'Cu': 8.96,
                  'Zn': 7.134, 'Ga': 5.907, 'Ge': 5.323, 'As': 5.776, 'Se': 4.809, 'Br': 3.122, 'Kr': 0.003733,
                  'Rb': 1.532, 'Sr': 2.64, 'Y': 4.469, 'Zr': 6.506, 'Nb': 8.57, 'Mo': 10.22, 'Tc': 11.5,
                  'Ru': 12.37,
                  'Rh': 12.41, 'Pd': 12.02, 'Ag': 10.501, 'Cd': 8.69, 'In': 7.31, 'Sn': 7.287, 'Sb': 6.685,
                  'Te': 6.232,
                  'I': 4.93, 'Xe': 0.005887, 'Cs': 1.873, 'Ba': 3.594, 'La': 6.145, 'Ce': 6.77, 'Pr': 6.773,
                  'Nd': 7.007,
                  'Pm': 7.26, 'Sm': 7.52, 'Eu': 5.244, 'Gd': 7.895, 'Tb': 8.23, 'Dy': 8.55, 'Ho': 8.795,
                  'Er': 9.066,
                  'Tm': 9.321, 'Yb': 6.965, 'Lu': 9.84, 'Hf': 13.31, 'Ta': 16.654, 'W': 19.25, 'Re': 21.02,
                  'Os': 22.61,
                  'Ir': 22.56, 'Pt': 21.46, 'Au': 19.282, 'Hg': 13.5336, 'Tl': 11.85, 'Pb': 11.342, 'Bi': 9.807,
                  'Po': 9.32, 'At': 'missing', 'Rn': 0.00973, 'Fr': 5, 'Ra': 10.07, 'Ac': 11.72, 'Th': 15.37,
                  'Pa': 19.1, 'U': 20.45, 'Np': 20.2, 'Pu': 13.67, 'Am': 13.5, 'Cm': 14.78, 'Bk': 15.1,
                  'Cf': 'missing',
                  'Es': 'missing', 'Fm': 'missing', 'Md': 'missing', 'No': 'missing', 'Lr': 'missing',
                  'Rf': 'missing',
                  'Db': 'missing', 'Sg': 'missing', 'Bh': 'missing', 'Hs': 'missing', 'Mt': 'missing',
                  'Ds': 'missing',
                  'Rg': 'missing', 'Cn': 'missing'}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class get_candidate(object):

    def __init__(self, args):
        
        self.downstream_save = downstream_to_lmdb(args.dowmstream_lmdb_save_path)
        self.composition_space_list = get_candidate.generate_composition_space(args.element_ranges)
        mean_std_path = " "
        mean_std = np.loadtxt(mean_std_path)
        self.stress_mean = mean_std[-1, 0]
        self.stress_std = mean_std[-1, 1]        
        self.physic_feature_mean = mean_std[:-2, 0]
        self.physic_feature_std = mean_std[:-2, 1]                
        self.procecssing_list = args. procecssing_list      
        self.physic_feature_list = args.physic_feature_list        
        self.homo = args.homo
        self.hot_roll = args.hot_roll
        self.anneal1 = args.anneal1
        self.encode_model = "steelbert"
     
    def obtain_data(self,  num_processes=30):
        
        data_save_list = []
        #print("the size of composition space：",len(self.composition_space_list))
        get_candidate.save_to_txt(self.composition_space_list, "./compositions.txt") 

        chunk_size = len(self.composition_space_list) // num_processes
        sub_dicts = [self.composition_space_list[i * chunk_size : (i + 1) * chunk_size] for i in range(num_processes)]

        if len(self.composition_space_list) % num_processes != 0:
            sub_dicts[-1].extend(self.composition_space_list[num_processes * chunk_size :])        
        #get_candidate.get_lmdb_data()        
        with multiprocessing.Manager() as manager:
            result = manager.list()
            processes = []
            for i in range(num_processes):
                process = multiprocessing.Process(
                    target=self.process_task, args=(sub_dicts[i], result)
                )
                processes.append(process)
                process.start()
            for process in processes:
                process.join()
            final_result = list(result)
            final_result = list(chain.from_iterable(final_result))               
        self.downstream_save.save_to_lmdb(final_result)
        
    
    def process_task(self, data, result):
        processed_data = self.get_lmdb_data(data)
        result.append(processed_data)


    def get_lmdb_data(self, composition_space_list):
        data_save_list = []
        for i in composition_space_list:
            name, element_list, composition_list = get_candidate.process_composition(i)
            physic_feature = Wdk_made_feature(element_list, composition_list, self.physic_feature_list, mole_fraction=False).get_features()
            physic_feature = np.array(physic_feature).reshape(1, -1)            
            physic_feature_nor = (physic_feature-self.physic_feature_mean)/self.physic_feature_std            
            text_list = []
            if self.procecssing_list["Homo_Temp/℃"] != 25:
                homo_params = {
                        "Homo_Temp/℃": self.procecssing_list["Homo_Temp/℃"],
                        "Homo_Time/h": self.procecssing_list["Homo_Time/h"]
                    }
                text_list.append(get_candidate.generate_processing(self.homo, **homo_params))       
                
            if self.procecssing_list["Roll_temp/℃"] != 25:
                roll_params = {
                        "Roll_temp/℃": self.procecssing_list["Roll_temp/℃"],
                        "Deform_rate(%)": self.procecssing_list["Deform_rate(%)"]
                    }
                text_list.append(get_candidate.generate_processing(self.hot_roll, **roll_params)) 
                
            if self.procecssing_list["Anneal_Temp_1/℃"] != 25:                
                anneal1_params = {
                        "Anneal_Temp_1/℃": self.procecssing_list["Anneal_Temp_1/℃"],
                        "Anneal_Time_1/h": self.procecssing_list["Anneal_Time_1/h"]
                    }
                text_list.append(get_candidate.generate_processing(self.anneal1, **anneal1_params))   
                
            if self.procecssing_list["Anneal_Temp_2/℃"] != 25:                
                anneal2_params = {
                        "Anneal_Temp_2/℃": self.procecssing_list["Anneal_Temp_2/℃"],
                        "Anneal_Time_2/h": self.procecssing_list["Anneal_Time_2/h"]
                    }
                text_list.append(get_candidate.generate_processing(anneal2, **anneal2_params))    
            text_composition = " ".join(text_list)
            #print("text_composition:", text_composition)
            language_model = LLM_process(self.encode_model)
            language_data = language_model.steelbert_method(element_list, composition_list, [text_composition])
            data_save_list.append((name, physic_feature_nor, language_data.cpu().numpy().reshape(-1, 1)))
        return data_save_list


    @staticmethod       
    def save_to_txt(compositions, filename):
        with open(filename, "w") as file:
            for comp in compositions:
                comp_str = ", ".join([f"{k}: {v}" for k, v in comp.items()])
                file.write(comp_str + "\n")

      
    
    @staticmethod
    def generate_composition_space(ranges):
        elements = list(ranges.keys())       
        combinations = itertools.product(*[ranges[elem] for elem in elements])        
        valid_compositions = deque()
        for combo in combinations:
            total = sum(combo)
            if total <= 100:
                nb_percent = 100 - total
                composition = {elem: percent for elem, percent in zip(elements, combo)}
                composition["Nb"] = round(nb_percent, 3)
                composition = {k: v for k, v in composition.items() if v != 0}                
                density = get_candidate.get_element_density(composition)
                if density > 9.5:      
                    continue
                w_percent = composition.get("W", 0)
                Mo_percent = composition.get("Mo", 0)                
                if w_percent == 6 and Mo_percent>1:
                    continue                    
                c_percent = composition.get("C", 0)
                n_percent = composition.get("N", 0)
                if c_percent  > 0.1:
                    continue                
                if c_percent + n_percent > 0.3:
                    continue                
                c_n_b_y_count = sum(1 for elem in ["C", "N", "B", "Y"] if elem in composition)
                if c_n_b_y_count > 3:
                    continue                
                if len(composition) >= 4:
                    valid_compositions.append(composition)                
        return list(valid_compositions)
    
    @staticmethod 
    def get_element_density(composition):
        w_e_list = []
        element_list = list(composition.keys())
        element_list = [x for x in element_list if x != "N"]            
        for i in element_list:
            weight_frac = composition[i]
            element_density = atomic_density[i]
            out = (weight_frac*0.01)/element_density
            w_e_list.append(out)
        return 1/sum(w_e_list)

    @staticmethod 
    def process_composition(composition):
        keys = list(composition.keys())
        values = list(composition.values())
        merged_str = "".join([f"{k}{v}" for k, v in composition.items()])        
        return merged_str, keys, values

    @staticmethod
    def generate_processing(template: str, **kwargs) -> str:

        return template.format(**kwargs)





class Test(object):

    def __init__(self, args):
    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model = Strain_Stress_Encoder(args).to(self.device)
    
        lmdb_curvle_dataset = LMBD_Downstream_Dataset(args.dowmstream_lmdb_save_path, 512)
             
       
        self.data_loader_curvle = DataLoader(lmdb_curvle_dataset, batch_size=1, shuffle=False)        
        mean_std_path = ""
        mean_std = np.loadtxt(mean_std_path)
        self.stress_mean = mean_std[-1, 0]
        self.stress_std = mean_std[-1, 1]

        strain_array = np.arange(0, 100, 0.004)
        
        strain_mean = mean_std[-2, 0]
        strain_std = mean_std[-2, 1]
        self.strain_array_nor = (strain_array-strain_mean)/strain_std
 
        self.predict_seg = 1
        
    def test(self, checkpoint_path):    
        print("Test start on {}:".format(self.device))         
        strain_array_segment_list, start_index_list = self.get_strain_input(self.strain_array_nor, num_segments=10) 
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval() 
        with torch.no_grad():            
            for name, physic_feature_input, language_input in self.data_loader_curvle:
                #print("name", name[0])
                curvle_list = []
                for strain, start_index in zip(strain_array_segment_list, start_index_list):
                    strain_input = torch.tensor(strain).to(dtype=torch.float32).to(self.device)
                    strain_input = strain_input.unsqueeze(1).unsqueeze(0)
                    physic_feature_input = physic_feature_input.to(self.device)
                    language_input = language_input.to(self.device)
                    outputs = self.model(physic_feature_input, strain_input, language_input, [start_index])  #[1, 512]                     
                    curvle_list.append(outputs.cpu().numpy())                
                self.process_and_save_data(curvle_list, output_file="{}.txt".format(name[0]))

    def process_and_save_data(self, data_list, output_file, downsample_factor=10):
        stacked_data = np.hstack(data_list)          
        denormalized_data = stacked_data * self.stress_std + self.stress_mean        
        denormalized_data = denormalized_data.reshape(-1, 1)
        downsampled_stress = denormalized_data[::downsample_factor]
        np.savetxt(output_file, downsampled_stress)        

      
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
        
        


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    train_process = get_candidate(args)
    train_process.obtain_data()
    downstream_predict = Test(args)
    downstream_predict.test(checkpoint_path=" ")













































