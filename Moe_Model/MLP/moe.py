import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from typing import Tuple
from Args import ModelArgs
from dataclasses import dataclass, field




class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
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
        """ 适合单个机器，单个GPU的版本(wdk20250507) """
        # 分数计算 (矩阵乘法优化)
        scores = torch.matmul(x, self.weight.t()) + self.bias  # [B, n_experts]

        # 激活函数处理
        if self.score_func == "softmax":
            probs = scores.softmax(dim=-1)
        else:  # sigmoid模式
            probs = torch.sigmoid(scores)

        # Top-K选择 
        weights, indices = torch.topk(probs, self.topk, dim=-1)

        # 权重归一化 (仅sigmoid需要)
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)

        return probs, indices

class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))



class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts  # 总专家数
        self.n_activated_experts = args.n_activated_experts

        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim)
                                      for _ in range(self.n_routed_experts)])  # 全量创建专家

        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)

        
        for expert_idx in range(self.n_routed_experts):
            mask = (indices == expert_idx).any(dim=1)
            if not mask.any():
                continue
           
            y[mask] += self.experts[expert_idx](x[mask]) * weights[mask].sum(dim=1, keepdim=True)

        z = self.shared_experts(x)
        
        return weights, indices, (y + z).view(shape)





if __name__ == "__main__":
    x = torch.randn((8, 1024, 2048))
    model = MoE(ModelArgs)
    weights, indices, output = model(x)
    print(output.shape)
    print(weights)
    print(indices)
    
    
    
    