import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from typing import Tuple, Literal
from dataclasses import dataclass, field



@dataclass
class ModelArgs:
    """
    moe参数    
    """    


   # moe参数
    dim: int=256
    n_routed_experts: int=4
    n_activated_experts: int=1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    moe_inter_dim: int=256
    n_shared_experts: int=1









class MLP(nn.Module):
    """
    简单的神经网络，不同的是激活函数选用silu。   
    """   
    def __init__(self, dim: int, inter_dim: int):

        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    Attributes:
        dim (int): 输入特征维度
        n_experts (int): 总专家数量
        topk (int): 激活的专家数
        score_func (str): 选择'softmax'或'sigmoid'
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_experts = args.n_routed_experts  # 使用全量专家数
        self.topk = args.n_activated_experts
        self.score_func = args.score_func

        # 简化参数初始化 (移除分组和条件判断)
        self.weight = nn.Parameter(torch.empty(self.n_experts, self.dim))
        self.bias = nn.Parameter(torch.empty(self.n_experts))  # 无条件使用偏置

        # 初始化参数
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 计算分数,其中self.weight.t()代表转置。x-->[-1, dim]=[4096, 256]
        scores = torch.matmul(x, self.weight.t()) + self.bias  # [4096, n_experts]

        # 激活函数处理
        if self.score_func == "softmax":
            probs = scores.softmax(dim=-1)
        else:  # sigmoid模式
            probs = torch.sigmoid(scores)
        #print("probs.shape:", probs.shape)   #[4096, 4] 4096 = batch_size*sqs_length = 8*512
        # Top-K选择 
        weights, indices = torch.topk(probs, self.topk, dim=-1)
        #print("weights", weights.shape, weights)  # [4096, 1]
        #print("indices", indices.shape, indices)  # [4096, 1]
        # 权重归一化 (仅sigmoid需要)
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)

        return weights, indices

class Expert(nn.Module):
    """
    定义专家模型，模型架构和MLP一致。
    """
    def __init__(self, dim: int, inter_dim: int):

        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts  # 总专家数
        self.n_activated_experts = args.n_activated_experts

        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim)
                                      for _ in range(self.n_routed_experts)])  # 创建全部专家

        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)  #创建共享专家

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        #print("weights1", weights.shape, weights)
        #print("indices1", indices.shape, indices)
        y = torch.zeros_like(x)

        
        for expert_idx in range(self.n_routed_experts):
            mask = (indices == expert_idx).any(dim=1)
            if not mask.any():
                continue
           
            y[mask] += self.experts[expert_idx](x[mask]) * weights[mask].sum(dim=1, keepdim=True)

        z = self.shared_experts(x)

        return (y + z).view(shape)



if __name__ == "__main__":
    x = torch.randn((8, 512, 256))
    model = MoE(ModelArgs)
    output = model(x)
    print(output.shape)