import sys
import os
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取 Nb_alloy 的绝对路径
# 假设 Nb_alloy 是 model.py 的上两级目录
nb_alloy_path = os.path.dirname(os.path.dirname(current_file_path))

# 将 Nb_alloy 添加到 sys.path
if nb_alloy_path not in sys.path:
    sys.path.insert(0, nb_alloy_path)
print(nb_alloy_path)

import numpy as np
import torch
import pandas as pd
import lmdb
from data.tools.data_trans_lmdb import data_to_lmdb, curvle_data_to_lmdb
from Args import args
from data.tools.utils import function



# 这里只需要导入所有训练数据的lmdb打包文件即可。
class strain_stress_data_process(object):
    """
    一些说明：
        训练用的应力应变曲线切分后保存在一个文件夹中。
    实现目标：
        将数据进行预处理(含标准化)，并把数据以lmdb格式元组的形式存放，该数据可以直接通过一个类(继承Dataset类)，加载给模型训练。
    Args:
        train_curvle_path (str): 所有应力应变曲线训练集所在的文件夹。
    Returns:
        用于训练的所有曲线的切片数据。

    """

    def __init__(self, args):
    
        self.train_all_curvle_path = args.all_curvle_path
    
        self.csv_path = args.csv_path
        self.extract_Paper_Num = args.extract_Paper_Num
        
        #segment保存的lmdb地址
        self.all_segment_train_data = data_to_lmdb(args.all_train_segment_lmdb_path)

        #导入实验数据
        self.exepriment_data = self._read_csv_data()
              
        #这里需要完成对应力应变曲线数据的加载以及预处理，以及其对应的物理特征和文本特征
    def get_data(self, normalize_save=True):
        """
        Args:
            normalize_save (bool):是否保存特征和应力应变的均值和标准差到本地。
        """
        #应力应变曲线数据导入（分成训练集和测试集, 其中训练集和测试集需要对physic feature和strain stress求均值和标准差。然后在保存为lmdb_data时对这些特征进行标准化）  
        curvle_train_tuples = function.obtain_data(curvle_path=self.train_all_curvle_path, exepriment_data=self.exepriment_data, element_name_list=args.element_name_list, physic_feature_list=args.physic_feature_list)
        #在这里对训练集添加高斯误差
        curvle_train_tuples = function.add_gaussian_noise(curvle_train_tuples, radio=0.01, n=5)
        
        mean_std_array=function.normalize(curvle_train_tuples, normalize_save=True)

        
        #对curvle_train_tuples和curvle_test_tuples进行归一化
        curvle_train_tuples = function.input_data_norm(curvle_train_tuples, mean_std_array)

        
        #根据不同的需求选择不同的处理方案并保存为lmdb格式。
        #保存为curvle形式
        #self.curvle_train_data.save_to_lmdb(curvle_train_tuples)

        
        # 保存为segment形式
        segment_train_tuples = function.trans_segment(curvle_train_tuples, args.split_length, args.overlap_length)
        self.all_segment_train_data.save_to_lmdb(segment_train_tuples)

       
    def _read_csv_data(self):
        #excell(.xlsx)数据导入及第一步预处理并取出指定编号的实验数据
        csv_data = pd.read_excel(self.csv_path, skiprows=1)
        csv_data = csv_data.iloc[:, 1:]
        exepriment_data = csv_data.loc[csv_data["Paper_Num"].isin(self.extract_Paper_Num)]
        return exepriment_data
        
        
if __name__=='__main__':

    data_process = strain_stress_data_process(args)
    data_process.get_data()



  
        

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
