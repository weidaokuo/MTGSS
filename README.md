# Data-driven Design of Nb-W Refractory Alloys Using Transformer-Based Stress-Strain Modeling

##  Project Overview
This repository contains the official implementation of our work on **Data-driven Design of Nb-W Refractory Alloys Using Transformer-Based Stress-Strain Modeling**. We propose a novel multimodal MTGSS (Multimodal Transformer-based Generative Stress-Strain) model architecture that synergistically integrates physics-informed descriptors with language-model-encoded alloy narratives. This framework enables the accurate generation of stress-strain curves for Nb-W alloys under specified compositions and processing conditions.
Below is the user guide.

---

## Environment & Dependencies

### Requirements
This project is developed in a PyTorch environment managed via Conda. The requirements.txt file specifies the exact versions of all modules used in this project.
Note: It is strongly recommended to run this project in a GPU-enabled CUDA environment. The author used an NVIDIA H100 80GB GPU with CUDA version 13.0.

### Installation
Create a virtual environment using conda 
```bash

conda create -n pytorch python=3.13
conda activate pytorch
```
Go to the website https://pytorch.org/get-started/locally/ to find the installation command corresponding to your CUDA version. Since our system is Linux with CUDA 13.0, the command to install PyTorch is:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
Next, install the dependencies required for this project using the following command:
```bash
pip install -r requirements.txt
```

**Special Note:**
We have developed a library named obtain_HEA_features for this project. The local installation package is located in the /package directory. You can install it using the following command:

```bash
pip install ./package/obtain_HEA_features-0.1.2-py3-none-any.whl
```

Also, our model employs the SteelBERT model to encode the composition and processing parameters of alloys. This model needs to be downloaded locally into the ./data/llm/steelbert folder for use. It can be downloaded from the following website: https://huggingface.co/MGE-LLMs/SteelBERT/tree/main. After downloading, run the command below to test the model:

```bash
python ./data/tools/LLM_encode.py
```

If no errors occur during the test, it indicates that the SteelBERT model can be loaded successfully. Otherwise, please check whether the model_path in the Model_Select class in the LLM_encode.py file is correctly defined, and verify that the model has been fully downloaded and placed in the ./data/llm/steelbert folder.


### Model Training
We have packaged the training data into LMDB files and stored them in the ./data/lmdb_data folder. Before starting training, please set the output_moe_weights parameter in Args.py to False. Then run the following command:

```bash
python main.py
```

### Prediction & Inference Example

We have also uploaded our trained weight file (checkpoint.pth) to the ./weights folder, allowing users to directly perform inference using our pre-trained model.

The inference procedure is as follows:

1. Add new alloy compositions and processing parameters to the end of the file data/origin_data/data.xlsx. Then specify the corresponding Paper_Num and Alloy_Num.  
   As an example for predicting a single alloy composition (as shown in the sample rows of data.xlsx), set Paper_Num = 14 and Alloy_Num = 0.

2. Configure prediction indices in Args.py:  
   Locate the variables predict_csv_Num and Alloy_Num_list.  
   - Set predict_csv_Num = 14.  
   - Set Alloy_Num_list to the range (0, 1) (left-closed, right-open interval, i.e., includes 0 only).  
   
   If you need to predict multiple alloys, simply append all new entries to the end of data.xlsx, keeping Paper_Num = 14 unchanged and incrementing Alloy_Num sequentially (e.g., 0, 1, 2, ..., m). Then set Alloy_Num_list = (0, m + 1).

3. Before running inference, ensure that the parameter output_moe_weights in Args.py is set to True. Then execute the following command:
4. 
```bash
python predict.py
```
### Results

The predicted stress-strain curves can be found in the results folder.  
For example, the file named Paper_Num_14_0.txt corresponds to the prediction for Paper_Num = 14 and Alloy_Num = 0, where the first column represents strain and the second column represents stress.  

Additionally, a corresponding image named Paper_Num_14_0.png is saved, allowing users to visually inspect the resulting stress-strain curve.



































