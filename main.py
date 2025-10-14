import sys
import os
current_file_path = os.path.abspath(__file__)
current_file_path = os.path.dirname(current_file_path)
sys.path.append(current_file_path)
from data.load_dataset import LMBD_Segment_Dataset, LMBD_Curvle_Dataset
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
#from torch.utils.tensorboard import SummaryWriter
from Args import args
from torch.cuda.amp import GradScaler, autocast
from concurrent.futures import ThreadPoolExecutor
save_executor = ThreadPoolExecutor(max_workers=8)
from Moe_Model.main_model import Strain_Stress_Encoder



train_order = "processing_inter"   



class Segment_train(object):

    def __init__(self, args):
    
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  
        self.model = Strain_Stress_Encoder(args).to(self.device)
        self.optimizer = torch.optim.AdamW(
                                             self.model.parameters()
                                            ,lr=args.learning_rate
                                            ,betas=(0.9, 0.999)
                                            ,eps = 1e-8
                                            ,weight_decay=0.01    
                                           )  

        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 200, 300], gamma=0.8, last_epoch=-1)       
        self.scheduler_warmup = LambdaLR(self.optimizer, lr_lambda = self.warmup_lr_lambda)   
        #self.scheduler_cosine = CosineAnnealingLR(self.optimizer, T_max=100 - args.pretrain_warmup_steps, eta_min=args.learning_rate * 0.1)        
        self.batch_size = args.segment_train_batch_size    
        self.epoch = args.pretrain_max_epochs  
        lmdb_dataset = LMBD_Segment_Dataset(args.all_train_segment_lmdb_path, 512)         
        self.data_loader = DataLoader(lmdb_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        self.mse_loss = nn.MSELoss(reduction='mean')

    def train(self, recurrent = False):
        loss_all = [] 
        mse_loss_list = []
        kv_loss_list = []
        diff_loss_list = []
        learning_rate = []
        print("Training start on {}:".format(self.device))        
        start_epoch=0
        if recurrent==True:
            checkpoint = torch.load('./weights/{}checkpoint_pretrain.pth'.format(train_order))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #self.scheduler_cosine.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        for epoch in range(start_epoch, args.pretrain_max_epochs): 
            
            loss_list = []
            loss_mse = []
            loss_kv = []
            loss_diff = []
            for physic_feature_input, strain_input, language_input, stress_output, start_index in self.data_loader:
                self.optimizer.zero_grad()
                physic_feature_input = physic_feature_input.to(self.device)
                strain_input = strain_input.to(self.device)
                language_input = language_input.to(self.device)
                stress_output = stress_output.to(self.device)
                start_index = start_index.to(self.device)                 
                outputs = self.model(physic_feature_input, strain_input, language_input, start_index) 
                loss1 = self.mse_loss(outputs, stress_output)
                loss2 = Segment_train.kl_divergence_loss(outputs, stress_output)
                loss3 = Segment_train.second_order_diff_loss(outputs, stress_output)
                total_loss = loss1 + 0.1*loss2 + 0.1*loss3   
                loss_list.append(total_loss.item())
                loss_mse.append(loss1.item())
                loss_kv.append(loss2.item())
                loss_diff.append(loss3.item())                    
                total_loss.backward()
                self.optimizer.step()
  
            loss_all.append(sum(loss_list) / len(loss_list))
            mse_loss_list.append(sum(loss_mse) / len(loss_mse))
            kv_loss_list.append(sum(loss_kv) / len(loss_kv))
            diff_loss_list.append(sum(loss_diff) / len(loss_diff))
            learning_rate.append(self.optimizer.param_groups[0]['lr'])
            
            if epoch < args.pretrain_warmup_steps:
                self.scheduler_warmup.step()
            #else:
            #    self.scheduler_cosine.step()         
            if epoch % 100 == 0:                   
                checkpoint = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    #"scheduler_state_dict": self.scheduler_cosine.state_dict(),
                    "epoch":epoch,
                }
                save_executor.submit(torch.save, checkpoint, f"./weights/{train_order}_checkpoint_pretrain{epoch}.pth")
                
                save_executor.submit(np.savetxt, f"./loss/{train_order}_train_loss_all{epoch}.txt", np.array(loss_all), delimiter=" ")
                
                save_executor.submit(np.savetxt, f"./loss/{train_order}_train_loss_mse{epoch}.txt", np.array(mse_loss_list), delimiter=" ")                

                save_executor.submit(np.savetxt, f"./loss/{train_order}_train_loss_kv{epoch}.txt", np.array(kv_loss_list), delimiter=" ")

                save_executor.submit(np.savetxt, f"./loss/{train_order}_train_loss_diff{epoch}.txt", np.array(diff_loss_list), delimiter=" ") 
                
                save_executor.submit(np.savetxt, f"./loss/{train_order}_learning_rate{epoch}.txt", np.array(learning_rate), delimiter=" ")
                
            print("0Epoch_all:{} loss:".format(epoch), sum(loss_list) / len(loss_list)) 
            
        np.savetxt("./loss/{}train_loss_all.txt".format(train_order), np.array(loss_all), delimiter=" ")  
        np.savetxt("./loss/{}train_loss_mse.txt".format(train_order), np.array(mse_loss_list), delimiter=" ") 
        np.savetxt("./loss/{}train_loss_kv.txt".format(train_order), np.array(kv_loss_list), delimiter=" ") 
        np.savetxt("./loss/{}train_loss_diff.txt".format(train_order), np.array(diff_loss_list), delimiter=" ") 


        

    def warmup_lr_lambda(self, current_step: int):
        if current_step < args.pretrain_warmup_steps:
            return float(current_step) / float(max(1, args.pretrain_warmup_steps))
        else:
            return 1
  
    @staticmethod        
    def kl_divergence_loss(predict, true):
        predict = F.softmax(predict, dim=-1)
        true = F.softmax(true, dim=-1)        
        loss = F.kl_div(predict.log(), true, reduction='batchmean')
        return loss

    @staticmethod 
    def second_order_diff_loss(pred, true):
        pred_curvature = torch.diff(pred, dim=-1, n=2)  
        true_curvature = torch.diff(true, dim=-1, n=2)  
        loss = torch.mean((pred_curvature - true_curvature)**2)
        return loss


if __name__ == '__main__':
    train_process = Segment_train(args)
    train_process.train(recurrent = False)














































