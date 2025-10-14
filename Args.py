from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal
import os
from typing import ClassVar


current_dir = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ModelArgs:

    
    output_moe_weights = True
    input_embedding_dim = 256   
    sqs_length: int=512
    

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


    dim: int = 256
    n_routed_experts: int=6
    n_activated_experts: int=2
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    moe_inter_dim: int=256
    n_shared_experts: int=1
    

    input_procross_dim: int=1536
  
    all_curvle_path = current_dir + " "    
    train_curvle_path = current_dir + " "
    test_curvle_path = current_dir + " "
    csv_path = current_dir + " "    
    
    extract_Paper_Num: list = field(default_factory=lambda:[13, 14])       
    predict_csv_Num: list = field(default_factory=lambda:[19])                                            
    Alloy_Num_list: list = field(default_factory=lambda:[i for i in range(97, 114)])                       
    
    
    
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
    
    # segment
    all_train_segment_lmdb_path = current_dir + " "           
    
    train_segment_lmdb_path = current_dir + " "    
    val_lmdb_path = current_dir + ""                    
    test_segment_lmdb_path = current_dir +  " "
    val_segment_ratio: int = 0.2   
    random_state: int = 1
    
    # curvle
    train_curvle_lmdb_path = current_dir + " "
    test_curvle_lmdb_path = current_dir + " "
    split_length: int = 512    
    overlap_length: int = 2          
    gauss_radio = 0.01               
    data_aug_n = 5                   
    
    # processing
    homo= "Homogenize at {Homo_Temp/℃}°C for {Homo_Time/h} hour."
    hot_roll = "Perform hot rolling at {Roll_temp/℃}°C with a rolling reduction of {Deform_rate(%)}."
    anneal1 = "Anneal the hot-rolled sample at {Anneal_Temp_1/℃}°C for {Anneal_Time_1/h} hour, followed by water quenching."
    anneal2 = "Perform a second annealing at {Anneal_Temp_2/℃}°C for {Anneal_Time_2/h} hour."
    
    
    # Training
    learning_rate=1e-5
    pretrain_warmup_steps=20
    pretrain_max_epochs=2000
    segment_train_batch_size = 16
    
    


    dowmstream_lmdb_save_path = current_dir + ""  
    
    # downstream element ranges
    element_ranges = {
        'W': [3, 4, 5, 6, 7, 8, 9, 10],
        'Mo': [0, 1, 2, 3],
        'Zr': [0, 1, 2, 3],
        'C': [0, 0.05, 0.1],  
        'Hf': [i * 0.5 for i in range(0, 21)],
        'Ta': [0, 1, 2, 3, 4, 5, 6, 7, 8],
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
  
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



