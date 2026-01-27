import numpy as np
import torch
import pandas as pd
from obtain_HEA_features import Wdk_made_feature
from data.tools.LLM_encode import LLM_process
import os
from scipy import interpolate
import random



homo= "Homogenize at {Homo_Temp/℃}°C for {Homo_Time/h} hour."
hot_roll = "Perform hot rolling at {Roll_temp/℃}°C with a rolling reduction of {Deform_rate(%)}."
anneal1 = "Anneal the hot-rolled sample at {Anneal_Temp_1/℃}°C for {Anneal_Time_1/h} hour, followed by water quenching."
anneal2 = "Perform a second annealing at {Anneal_Temp_2/℃}°C for {Anneal_Time_2/h} hour."





class function:
    @staticmethod
    def obtain_data(curvle_path, exepriment_data, element_name_list, physic_feature_list,  encode_model = "steelbert"):

        data_save_list = []
        all_alloy_nums_set = set(exepriment_data['Alloy_Num'])
        if os.path.exists(curvle_path):
            print("Path exists.")
        else:
            print("Path does not exist:", curvle_path)
        stress_curvle_path_list = os.listdir(curvle_path)
        
        for i in stress_curvle_path_list:
            ss_path = curvle_path + "/" + i
            ss_curvle = np.array(pd.read_csv(ss_path))   
            sorted_indices = np.argsort(ss_curvle[:, 0])[::-1]   
            ss_curvle = ss_curvle[sorted_indices]
            if len(ss_curvle)!=0:
                strain_array = np.arange(0, 100, 0.004)  
                x_data = ss_curvle[:, 0]
                y_data = ss_curvle[:, 1] 
                if x_data[0] != 0 or y_data[0] != 0:
                    x_data = np.insert(x_data, 0, 0)
                    y_data = np.insert(y_data, 0, 0)
                    f = interpolate.interp1d(x_data, y_data, kind='linear')
                    max_strain = x_data.max()
                    strain_new = strain_array[strain_array<=max_strain]
                    stress_new = f(strain_new)
                    strain_stress = np.hstack([strain_new.reshape(-1,1), stress_new.reshape(-1,1)])

            Paper_Num = i.split(".")[0]
            Alloy_Num = i.split(".")[1]
            if int(Alloy_Num) not in all_alloy_nums_set:
                continue  
            print("Paper_Num", Paper_Num)
            print("Alloy_Num", Alloy_Num)                        
            selected_rows1 = exepriment_data[exepriment_data["Paper_Num"].astype(str) == Paper_Num]
            feature_data = selected_rows1[selected_rows1["Alloy_Num"].astype(str) == Alloy_Num]
            rows = feature_data.loc[:, element_name_list]            
            element_list, composition_list = function.extract_non_zero(rows.iloc[0])
            physic_feature = Wdk_made_feature(element_list, composition_list, physic_feature_list, mole_fraction=False).get_features()
            try:
                function.check_list_for_invalid_values(physic_feature)
            except ValueError as e:
                print(e)  
                sys.exit(1)
            text_list = []
            if feature_data["Homo_Temp/℃"].item() != 25:
                homo_params = {
                        "Homo_Temp/℃": feature_data["Homo_Temp/℃"].item(),
                        "Homo_Time/h": feature_data["Homo_Time/h"].item()
                    }
                text_list.append(function.generate_composition(homo, **homo_params))       
                
            if feature_data["Roll_temp/℃"].item() != 25:
                roll_params = {
                        "Roll_temp/℃": feature_data["Roll_temp/℃"].item(),
                        "Deform_rate(%)": feature_data["Deform_rate(%)"].item()
                    }
                text_list.append(function.generate_composition(hot_roll, **roll_params)) 
                
            if feature_data["Anneal_Temp_1/℃"].item() != 25:                
                anneal1_params = {
                        "Anneal_Temp_1/℃": feature_data["Anneal_Temp_1/℃"].item(),
                        "Anneal_Time_1/h": feature_data["Anneal_Time_1/h"].item()
                    }
                text_list.append(function.generate_composition(anneal1, **anneal1_params))   
                
            if feature_data["Anneal_Temp_2/℃"].item() != 25:                
                anneal2_params = {
                        "Anneal_Temp_2/℃": feature_data["Anneal_Temp_2/℃"].item(),
                        "Anneal_Time_2/h": feature_data["Anneal_Time_2/h"].item()
                    }
                text_list.append(function.generate_composition(anneal2, **anneal2_params))
                
            text_composition = " ".join(text_list)                             
            language_model = LLM_process(encode_model)
            language_data = language_model.steelbert_method(element_list, composition_list, [text_composition])
            data_save_list.append((np.array(physic_feature).reshape(1, -1), language_data.cpu().numpy().reshape(-1, 1), strain_stress))
        return data_save_list
        
        
    @staticmethod
    def csv_predict_data(Paper_Num, Alloy_Num_list, exepriment_data, element_name_list, physic_feature_list,  encode_model = "steelbert"):

        data_save_list = []
 
        for Alloy_Num in Alloy_Num_list:
            print("Paper_Num", Paper_Num)
            print("Alloy_Num", Alloy_Num)            
            selected_rows1 = exepriment_data[exepriment_data["Paper_Num"].astype(int) == Paper_Num[0]]
            #print("selected_rows1:", selected_rows1)
            feature_data = selected_rows1[selected_rows1["Alloy_Num"].astype(int) == Alloy_Num]
            rows = feature_data.loc[:, element_name_list]
            element_list, composition_list = function.extract_non_zero(rows.iloc[0])           
            physic_feature = Wdk_made_feature(element_list, composition_list, physic_feature_list, mole_fraction=False).get_features()

            try:
                function.check_list_for_invalid_values(physic_feature)
            except ValueError as e:
                print(e)  
                sys.exit(1)
            text_list = []
            if feature_data["Homo_Temp/℃"].item() != 25:
                homo_params = {
                        "Homo_Temp/℃": feature_data["Homo_Temp/℃"].item(),
                        "Homo_Time/h": feature_data["Homo_Time/h"].item()
                    }
                text_list.append(function.generate_composition(homo, **homo_params))       
                
            if feature_data["Roll_temp/℃"].item() != 25:
                roll_params = {
                        "Roll_temp/℃": feature_data["Roll_temp/℃"].item(),
                        "Deform_rate(%)": feature_data["Deform_rate(%)"].item()
                    }
                text_list.append(function.generate_composition(hot_roll, **roll_params)) 
                
            if feature_data["Anneal_Temp_1/℃"].item() != 25:                
                anneal1_params = {
                        "Anneal_Temp_1/℃": feature_data["Anneal_Temp_1/℃"].item(),
                        "Anneal_Time_1/h": feature_data["Anneal_Time_1/h"].item()
                    }
                text_list.append(function.generate_composition(anneal1, **anneal1_params))   
                
            if feature_data["Anneal_Temp_2/℃"].item() != 25:                
                anneal2_params = {
                        "Anneal_Temp_2/℃": feature_data["Anneal_Temp_2/℃"].item(),
                        "Anneal_Time_2/h": feature_data["Anneal_Time_2/h"].item()
                    }
                text_list.append(function.generate_composition(anneal2, **anneal2_params))
                
            text_composition = " ".join(text_list)                                     
            language_model = LLM_process(encode_model)
            language_data = language_model.steelbert_method(element_list, composition_list, [text_composition])
            data_save_list.append((np.array(physic_feature).reshape(1, -1), language_data.cpu().numpy().reshape(-1, 1), Alloy_Num))
        return data_save_list    
        

    @staticmethod
    def curve_split(strain_stress_curve, split_length, overlap_length, save_start_index=True):
        """
       Split the stress-strain curve into multiple segments and save the index of the first value of each segment in the original sequence.
        """
        n = strain_stress_curve.shape[0]  
        segments = []  
        step = split_length - overlap_length  

        # Check if the input parameters are valid.
        if split_length <= 0 or overlap_length < 0 or split_length <= overlap_length:
            raise ValueError("split_length must be greater than 0, and split_length must be greater than overlap_length.")
        if n < split_length:
            raise ValueError("The curve length must be greater than or equal to the split_length.")

        start = 0
        while start + split_length <= n:
            segment = strain_stress_curve[start:start + split_length]
            if save_start_index == True:
                segments.append((segment, start))
            else:
                segments.append(segment)
            start += step

        # Process the last segment
        if start < n:
            segment = strain_stress_curve[-split_length:]
            if save_start_index == True:
                segments.append((segment, n - split_length))
            else:
                segments.append(segment)

        return segments


    @staticmethod   
    def trans_segment(curvle_tuples, split_length, overlap_length):
        
        segment_tuples = []
        for physic_feature_input, language_input, strain_stress_curvle in curvle_tuples:
            #print("len(strain_stress_curvle)", len(strain_stress_curvle), strain_stress_curvle.shape)
            curvle_split_datalist = function.curve_split(strain_stress_curvle, split_length, overlap_length, save_start_index=True)
            for i in curvle_split_datalist:
                segment_curvle = i[0]
                segment_tuples.append((physic_feature_input, segment_curvle[:,0], language_input, segment_curvle[:,1], i[1]))

        return segment_tuples
                        
        
    @staticmethod
    def extract_non_zero(row):
        """
        Define a function to extract the column names and values (non-zero) from a specific row of the series.
        """

        non_zero = row[row != 0]
        return non_zero.index.tolist(), non_zero.tolist() 

        return results[0][0], results[0][1]    
        


    @staticmethod
    def normalize(curvle_train_tuples, normalize_save=True):
        """
        Calculate the mean and standard deviation of the physical features and curves.
        """
        
        strain_stress_curve = []
        feature_data_list = []
        for i in curvle_train_tuples:
            feature_data_list.append(i[0])
            strain_stress_curve.append(i[2])
            
        # Merge all physical features
        physic_feature_array = np.vstack(feature_data_list)
        strain_stress_array = np.vstack(strain_stress_curve)      
        feature_mean = np.mean(physic_feature_array, axis=0)
        strain_stress_mean = np.mean(strain_stress_array, axis=0)
        
        # Calculate the standard deviation
        feature_std = np.std(physic_feature_array, axis=0)  
        strain_stress_std = np.std(strain_stress_array, axis=0)
        feature_mean_std = np.column_stack((feature_mean, feature_std))
        strain_stress_mean_std = np.column_stack((strain_stress_mean, strain_stress_std))
        mean_std_array = np.vstack([feature_mean_std, strain_stress_mean_std])
        if normalize_save:
            np.savetxt("./mean_std.txt", mean_std_array)
        
        return mean_std_array
        
       
    @staticmethod      
    def input_data_norm(curvle_train_tuples, mean_std_array):
        """
        curvle_train_tuples:  list, [(physic_feature, language_data, strain_stress),......]。
                physic_feature: (1, num_physic)
                language_data: (n, 1)
                strain_stress: (m, 2)
        mean_std_array: numpy array , shape[n, 2]
                
        """

        curvle_train_tuples_norm = []
        for physic_feature, language_data, strain_stress in curvle_train_tuples:
            # Standardized physical features
            n = physic_feature.shape[1]  
            physic_feature_norm = (physic_feature - mean_std_array[:n, 0]) / mean_std_array[:n, 1]           
            strain_norm = (strain_stress[:, 0] - mean_std_array[n, 0]) / mean_std_array[n, 1]
            stress_norm = (strain_stress[:, 1] - mean_std_array[n+1, 0]) / mean_std_array[n+1, 1]
            strain_stress_norm = np.column_stack((strain_norm, stress_norm))            
            curvle_train_tuples_norm.append((physic_feature_norm, language_data, strain_stress_norm))
        
        return curvle_train_tuples_norm    



    @staticmethod
    def generate_composition(template: str, **kwargs) -> str:

        return template.format(**kwargs)
        
    @staticmethod
    def check_list_for_invalid_values(data):
        invalid_indices = []
        
        for index, value in enumerate(data):
            if not isinstance(value, (int, float)):
                invalid_indices.append((index, "Non-numeric type"))
            elif isinstance(value, float) and np.isnan(value):
                invalid_indices.append((index, "NaN"))       
        if invalid_indices:
            raise ValueError(f"There are invalid values in the list. Please check if the alloy composition is correct：{invalid_indices}")
            sys.exit(1)  
            

   
    @staticmethod    
    def add_gaussian_noise(curvle_train_tuples, radio=0.01, n=5):
        new_curvle_train_tuples = []
        
        for data in curvle_train_tuples:
            physic_feature, language_data, strain_stress = data
            new_data_ori = (physic_feature, language_data, strain_stress)
            new_curvle_train_tuples.append(new_data_ori)
            for _ in range(n):
                noisy_strain_stress = strain_stress.copy()                
                second_column = noisy_strain_stress[:, 1]
                noise = np.random.normal(loc=0, scale=radio * np.abs(second_column))
                noisy_strain_stress[:, 1] = second_column + noise
                new_data = (physic_feature.copy(), language_data.copy(), noisy_strain_stress)
                new_curvle_train_tuples.append(new_data)
        
        return new_curvle_train_tuples    










































 
    
    
    
    