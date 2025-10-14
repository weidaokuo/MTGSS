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



    
def extract_non_zero(row):
    """
    定义一个函数，提取某一行series中非零的列名称和数值
    """
    results = []
    for index, row in df.iterrows():
        non_zero = row[row != 0.0]  # 提取非零元素
        element_list = non_zero.index.tolist()  # 非零元素的列名
        composition_list = non_zero.tolist()  # 非零元素的值
        results.append((element_list, composition_list))
    #non_zero = row[row != 0]
    #return non_zero.index.tolist(), non_zero.tolist() 
    if len(results) >1:
        raise ValueError("row 必须是一个单行 DataFrame")
    return results[0][0], results[0][1]



class function:
    @staticmethod
    def obtain_data(curvle_path, exepriment_data, element_name_list, physic_feature_list,  encode_model = "steelbert"):
        """       
        output:[(physic_feature, language_data, strain_stress),...]
            physic_feature: (1, num_physic)
            language_data: (1536, 1)
            strain_stress: (1200, 2)
        """
        data_save_list = []
        if os.path.exists(curvle_path):
            print("Path exists.")
        else:
            print("Path does not exist:", curvle_path)
        stress_curvle_path_list = os.listdir(curvle_path)
        
        for i in stress_curvle_path_list:
            #应力应变曲线数据的导入及初步处理（还需要预先求的所有曲线的均值和标准差）
            ss_path = curvle_path + "/" + i
            ss_curvle = np.array(pd.read_csv(ss_path))   #shape must be [n, 2]
            sorted_indices = np.argsort(ss_curvle[:, 0])[::-1]   #第一排排序
            ss_curvle = ss_curvle[sorted_indices]
            if len(ss_curvle)!=0:
                strain_array = np.arange(0, 100, 0.004)  #每500个点代表的strain片段长度是2%。
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
                   
            #print("strain_stress length:",len(strain_stress))
            #print(ss_path)
            #取出应变应变曲线和特征的对应行
            Paper_Num = i.split(".")[0]
            Alloy_Num = i.split(".")[1]
            #print("exepriment_data", exepriment_data)
            print("Paper_Num", Paper_Num)
            print("Alloy_Num", Alloy_Num)
            selected_rows1 = exepriment_data[exepriment_data["Paper_Num"].astype(str) == Paper_Num]
            #print("selected_rows1:", selected_rows1)
            feature_data = selected_rows1[selected_rows1["Alloy_Num"].astype(str) == Alloy_Num]
            #print("feature_data:", feature_data)
            #print("feature_data.loc[:, element_name_list]:",feature_data.loc[:, element_name_list])
            rows = feature_data.loc[:, element_name_list]
            #print(rows, len(rows), type(rows))
            # 将结果拆分为 element_list 和 composition_list
            #print("paper_num:", Paper_Num)
            #print("Alloy_Num:", Alloy_Num)
            #print("rows:", rows)
            element_list, composition_list = function.extract_non_zero(rows.iloc[0])
            
            #通过Wdk_made_feature获取对应element_list, composition_list涉及到的物理特征（注意physic_feature需要进行归一化，计算均值和标准差）
            physic_feature = Wdk_made_feature(element_list, composition_list, physic_feature_list, mole_fraction=False).get_features()
            #print("physic_feature:", physic_feature)
            try:
                function.check_list_for_invalid_values(physic_feature)
            except ValueError as e:
                print(e)  
                sys.exit(1)
            """
            text_composition需要修改为合适的,上面Wdk_made_feature需要完善后使用pip install之后导入
            """
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
            
            
             
            #text_composition =  generate_composition(template, dict)     #通过模板形式导入                     
                                        
                                        
            language_model = LLM_process(encode_model)
            language_data = language_model.steelbert_method(element_list, composition_list, [text_composition])
            data_save_list.append((np.array(physic_feature).reshape(1, -1), language_data.cpu().numpy().reshape(-1, 1), strain_stress))
            #print("physic_feature.shape:",np.array(physic_feature).reshape(1, -1).shape)
            #print("language_data.shape:",language_data.cpu().numpy().reshape(-1, 1).shape)
            #print("strain_stress.shape:",strain_stress.shape)
        return data_save_list
        
    @staticmethod
    def csv_predict_data(Paper_Num, Alloy_Num_list, exepriment_data, element_name_list, physic_feature_list,  encode_model = "steelbert"):
        """       
        output:[(physic_feature, language_data, strain_stress),...]
            physic_feature: (1, num_physic)
            language_data: (1536, 1)
            strain_stress: (1200, 2)
        """
        data_save_list = []
 
        for Alloy_Num in Alloy_Num_list:
 
            #print("exepriment_data", exepriment_data)
            print("Paper_Num", Paper_Num)
            print("Alloy_Num", Alloy_Num)
            
            selected_rows1 = exepriment_data[exepriment_data["Paper_Num"].astype(int) == Paper_Num[0]]
            #print("selected_rows1:", selected_rows1)
            feature_data = selected_rows1[selected_rows1["Alloy_Num"].astype(int) == Alloy_Num]
            #print("feature_data:", feature_data)
            #print("feature_data.loc[:, element_name_list]:",feature_data.loc[:, element_name_list])
            rows = feature_data.loc[:, element_name_list]
            #print(rows, len(rows), type(rows))
            # 将结果拆分为 element_list 和 composition_list
            element_list, composition_list = function.extract_non_zero(rows.iloc[0])
            
            #通过Wdk_made_feature获取对应element_list, composition_list涉及到的物理特征（注意physic_feature需要进行归一化，计算均值和标准差）
            physic_feature = Wdk_made_feature(element_list, composition_list, physic_feature_list, mole_fraction=False).get_features()
            #print("physic_feature:", physic_feature)
            try:
                function.check_list_for_invalid_values(physic_feature)
            except ValueError as e:
                print(e)  
                sys.exit(1)
            """
            text_composition需要修改为合适的,上面Wdk_made_feature需要完善后使用pip install之后导入
            """
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
            
            #name = "Paper_Num is {} and Alloy_Num is {}".format(Paper_Num, Alloy_Num)
             
            #text_composition =  generate_composition(template, dict)     #通过模板形式导入                     
                                        
                                        
            language_model = LLM_process(encode_model)
            language_data = language_model.steelbert_method(element_list, composition_list, [text_composition])
            data_save_list.append((np.array(physic_feature).reshape(1, -1), language_data.cpu().numpy().reshape(-1, 1), Alloy_Num))
            #print("physic_feature.shape:",np.array(physic_feature).reshape(1, -1).shape)
            #print("language_data.shape:",language_data.cpu().numpy().reshape(-1, 1).shape)
            #print("strain_stress.shape:",strain_stress.shape)
        return data_save_list    

    @staticmethod
    def curve_split(strain_stress_curve, split_length, overlap_length, save_start_index=True):
        """
        将应力应变曲线切分为多个片段，并保存每个片段的第一个值在原序列中的索引。

        参数:
            strain_stress_curve (np.ndarray): 形状为 [n, 2] 的 numpy 矩阵，表示应力应变曲线。
            split_length (int): 每个切分片段的固定长度。
            overlap_length (int): 切分时相邻两个片段之间的重叠长度。
        
        返回:
            list: 包含 (segment, start_index) 元组的列表，其中 segment 是曲线片段，start_index 是该片段的第一个值在原序列中的索引。
        """
        n = strain_stress_curve.shape[0]  # 曲线总长度
        segments = []  # 保存所有片段
        step = split_length - overlap_length  # 每次切分的步长

        # 检查输入参数是否合法
        if split_length <= 0 or overlap_length < 0 or split_length <= overlap_length:
            raise ValueError("split_length 必须大于 0，且 split_length 必须大于 overlap_length。")
        if n < split_length:
            raise ValueError("曲线长度必须大于或等于 split_length。")

        # 从前往后切分
        start = 0
        while start + split_length <= n:
            segment = strain_stress_curve[start:start + split_length]
            if save_start_index == True:
                segments.append((segment, start))
            else:
                segments.append(segment)
            start += step

        # 处理最后一个片段
        if start < n:
            # 从后往前取 split_length 长度的片段
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
    def split_list(tuples_list, ratio=0.8, random_state=1):
        """
        将包含 tuple 的 list 随机分成两份。

        参数:
        - tuples_list: list, 包含多个 tuple 的列表。
        - ratio: float, 第一份的占比，默认是 0.8。

        返回:
        - part1: list, 占比为 ratio 的部分。
        - part2: list, 占比为 1 - ratio 的部分。
        """
            # 设置随机种子
        if random_state is not None:
            random.seed(random_state)
        # 打乱列表顺序
        random.shuffle(tuples_list)

        # 计算分割点
        split_index = int(len(tuples_list) * ratio)

        # 分割列表
        part1 = tuples_list[:split_index]
        part2 = tuples_list[split_index:]

        return part1, part2  
        

        
    @staticmethod
    def extract_non_zero(row):
        """
        定义一个函数，提取某一行series中非零的列名称和数值
        """

        non_zero = row[row != 0]
        return non_zero.index.tolist(), non_zero.tolist() 

        return results[0][0], results[0][1]    
        


    #求物理特征和应力应变曲线的均值和标准差
    @staticmethod
    def normalize(curvle_train_tuples, normalize_save=True):
        """
        根据模板和工艺参数生成文本描述。

        Args:
            curvle_train_tuples: list, [(physic_feature, language_data, strain_stress),......]。
                physic_feature: (1, num_physic)
                language_data: (1536, 1)
                strain_stress: (1200, 2)
            normalize_save: bool。是否保存到本地。默认保存。

        Returns:
            str: 生成的文本描述。
        """
        strain_stress_curve = []
        feature_data_list = []
        for i in curvle_train_tuples:
            feature_data_list.append(i[0])
            strain_stress_curve.append(i[2])
        #堆叠所有physic_feature
        physic_feature_array = np.vstack(feature_data_list)
        strain_stress_array = np.vstack(strain_stress_curve)
        #这里写成按照特征名字存储的均值和标准差：arg.physic_feature_list查看所有特征名称(不太需要名称)。        
        feature_mean = np.mean(physic_feature_array, axis=0)
        strain_stress_mean = np.mean(strain_stress_array, axis=0)
        # 计算每列的标准差
        feature_std = np.std(physic_feature_array, axis=0)  
        strain_stress_std = np.std(strain_stress_array, axis=0)
        # 将均值和标准差合并为一个 [m, 2] 的矩阵
        feature_mean_std = np.column_stack((feature_mean, feature_std))
        strain_stress_mean_std = np.column_stack((strain_stress_mean, strain_stress_std))
        mean_std_array = np.vstack([feature_mean_std, strain_stress_mean_std])
        if normalize_save:
            np.savetxt("./0924_include_paper_curvle_processing_last_inter_mean_std.txt", mean_std_array)
        
        return mean_std_array
        
       
    @staticmethod      
    def input_data_norm(curvle_train_tuples, mean_std_array):
        """
        curvle_train_tuples:  list, [(physic_feature, language_data, strain_stress),......]。
                physic_feature: (1, num_physic)
                language_data: (1536, 1)
                strain_stress: (1200, 2)
        mean_std_array: numpy array , shape[n, 2]
                
        """
        # 初始化标准化后的列表
        curvle_train_tuples_norm = []
        
        # 遍历每个元组
        for physic_feature, language_data, strain_stress in curvle_train_tuples:
            # 标准化 physic_feature
            n = physic_feature.shape[1]  # 特征数量
            physic_feature_norm = (physic_feature - mean_std_array[:n, 0]) / mean_std_array[:n, 1]
            
            # 标准化 strain_stress
            strain_norm = (strain_stress[:, 0] - mean_std_array[n, 0]) / mean_std_array[n, 1]
            stress_norm = (strain_stress[:, 1] - mean_std_array[n+1, 0]) / mean_std_array[n+1, 1]
            strain_stress_norm = np.column_stack((strain_norm, stress_norm))
            
            # 将标准化后的结果添加到新列表中
            curvle_train_tuples_norm.append((physic_feature_norm, language_data, strain_stress_norm))
        
        return curvle_train_tuples_norm    



    @staticmethod
    def generate_composition(template: str, **kwargs) -> str:
        """根据模板和工艺参数生成文本描述。

        Args:
            template (str): 文本模板，使用命名占位符，例如 `{Anneal_Temp_1}`。
            **kwargs: 工艺参数，键值对形式。

        Returns:
            str: 生成的文本描述。
            
            
        # 示例使用
        template = "The processing of high entropy alloy is Al{Anneal_Temp_1/℃}Co{Anneal_Time_1/h}Cr{Quench_1}Fe{Anneal_Temp_2/℃}Ni{Anneal_Time_2/h}Ti."
        process_params = {
            "Anneal_Temp_1/℃": 800,
            "Anneal_Time_1/h": 2,
            "Quench_1": "water",
            "Anneal_Temp_2/℃": 600,
            "Anneal_Time_2/h": 1,
        }

        # 生成文本描述
        text_composition = generate_composition(template, **process_params)
        print(text_composition)   
        # 输出为：The processing of high entropy alloy is Al800Co2CrwaterFe600Ni1Ti{}.
        """
        return template.format(**kwargs)
        
    @staticmethod
    def check_list_for_invalid_values(data):
        invalid_indices = []
        
        for index, value in enumerate(data):
            # 检查是否为非数值类型
            if not isinstance(value, (int, float)):
                invalid_indices.append((index, "非数值类型"))
            # 检查是否为 NaN
            elif isinstance(value, float) and np.isnan(value):
                invalid_indices.append((index, "NaN"))
        
        # 如果发现无效值，报错并返回索引
        if invalid_indices:
            raise ValueError(f"列表中存在无效值,请检查合金成分是否正确：{invalid_indices}")
            sys.exit(1)  # 终止程序
            

               
                
    @staticmethod    
    def add_gaussian_noise(curvle_train_tuples, radio=0.01, n=5):
        new_curvle_train_tuples = []
        
        for data in curvle_train_tuples:
            physic_feature, language_data, strain_stress = data
            new_data_ori = (physic_feature, language_data, strain_stress)
            new_curvle_train_tuples.append(new_data_ori)
            for _ in range(n):
                # 复制 strain_stress 以避免修改原始数据
                noisy_strain_stress = strain_stress.copy()
                
                # 对第二列添加高斯误差
                second_column = noisy_strain_stress[:, 1]
                noise = np.random.normal(loc=0, scale=radio * np.abs(second_column))  # 2% 高斯误差
                noisy_strain_stress[:, 1] = second_column + noise
                
                # 创建新的数据元组
                new_data = (physic_feature.copy(), language_data.copy(), noisy_strain_stress)
                new_curvle_train_tuples.append(new_data)
        
        return new_curvle_train_tuples    










































 
    
    
    
    