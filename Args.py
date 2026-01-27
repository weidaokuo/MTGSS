from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal
import os
from typing import ClassVar

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
#print(current_dir)

@dataclass
class ModelArgs:
    """
    Summary of project parameters
    """    
    # output_moe_weights = False
    output_moe_weights = True
    
    # Parameters of Encoder_layer
    input_embedding_dim = 256   
    sqs_length: int=512
    
    # Model parameter configuration
    HEA_input_dim = 15
    HEA_output_dim = 128
    strain_input_dim = 1
    strain_output_dim = 128        
    d_model:int = 256                   
    ffn_hidden:int = 256
    drop_prob:float = 0.1
    n_head:int = 8
    n_layers: int = 6
    input_d_model:int = 256
    max_len:int=50000    
    output_d_model:int=1
    input_procross_dim: int=1536
    
   # MoE parameter configuration
    dim: int = 256
    n_routed_experts: int=6
    n_activated_experts: int=2
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    moe_inter_dim: int=256
    n_shared_experts: int=1
    

    all_curvle_path = current_dir + "/data/origin_data/curvle_data"               
    csv_path = current_dir + "/data/origin_data/data.xlsx"                          
    extract_Paper_Num: list = field(default_factory=lambda:[13])  
    
    # The Paper_Num and Alloy_Num indices in the data.xlsx file used for downstream task evaluation (with Alloy_Num_list being left-open and right-closed).
    predict_csv_Num: list = field(default_factory=lambda:[14])      
    Alloy_Num_list: list = field(default_factory=lambda:[i for i in range(0, 1)])   
        
    # Save path for the LMDB-formatted files
    all_train_segment_lmdb_path = current_dir + "/data/lmdb_data"      
    
    composition_start_index: int = 3
    process_name_list: list = field(default_factory=lambda:["Anneal_Temp_1/℃", "Anneal_Time_1/h", "Quench_1", "Anneal_Temp_2/℃", "Anneal_Time_2/h"])
    element_name_list: list = field(default_factory=lambda:["Nb", "W", "Mo", "Zr", "C", "Hf", "Ti", "Ta", "Si", "N", "V", "Sc", "Ru", "B", "Y"])
    physic_feature_list: list = field(default_factory=lambda:[
                              "vec", "cohesive_energy",  "average_electronegativity",
                              "electronegativity_difference","average_atomic_size", "atomic_size_difference",
                              "mixed_entropy","mixed_enthalpy", "Tm", "density", "melting_enthalpy",
                              "thermal_conductivity", "specific_heat", "lattice_constant",
                                "omega"
                              ])
   
    random_state: int = 1
    split_length: int = 512    
    overlap_length: int = 2          
    gauss_radio = 0.01               
    data_aug_n = 5                   
    
    # An example for the process input section; modifications should be made in /data/tools/utils.py.
    homo= "Homogenize at {Homo_Temp/℃}°C for {Homo_Time/h} hour."
    hot_roll = "Perform hot rolling at {Roll_temp/℃}°C with a rolling reduction of {Deform_rate(%)}."
    anneal1 = "Anneal the hot-rolled sample at {Anneal_Temp_1/℃}°C for {Anneal_Time_1/h} hour, followed by water quenching."
    anneal2 = "Perform a second annealing at {Anneal_Temp_2/℃}°C for {Anneal_Time_2/h} hour."
    
    
    #训练过程中的参数
    learning_rate=1e-5
    pretrain_warmup_steps=20
    pretrain_max_epochs=601
    segment_train_batch_size = 8
    
    

    #获取候选空间相关参数
    #保存地址
    dowmstream_lmdb_save_path = current_dir + "/data/lmdb_data/downstream_path/411"  
    
    #元素空间
    element_ranges = {
        'W': [6, 7, 8, 9, 10],
        'Mo': [0, 1, 2, 3],
        'Zr': [0, 1, 2, 3],
        'C': [0, 0.05, 0.1],  
        'Hf': [2, 4, 6, 8, 10],
        'Ta': [0, 1, 2, 3, 4, 5, 6],
        'N': [0, 0.1, 0.2], 
        'Y': [0, 0.1, 0.2, 0.3]
    }
    
 
    
    procecssing_list = {
            "Homo_Temp/℃": 1800,
            "Homo_Time/h": 2,
            "Roll_temp/℃": 450,
            "Deform_rate(%)": 85,
            "Anneal_Temp_1/℃": 1450,
            "Anneal_Time_1/h": 1,
            "Anneal_Temp_2/℃" : 25,
            "Anneal_Time_2/h" : 0
                                } 
    
args = ModelArgs()    
#print(args.train_curvle_path)    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



