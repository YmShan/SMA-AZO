import sys
import torch
import torch.nn as nn
import torchvision
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
sys.path.append("../..")
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, model, layer, monitor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda import amp
import os
import random
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict


class SMA(nn.Module):
    def __init__(self, channel=512, time=16, kernels=[1, 3, 5, 7], C_reduction=16, T_reduction=4, AZO=False, RCR=4, RTR=4):
        super().__init__()
        self.AZO = AZO
        self.RCR = RCR
        self.RTR = RTR
        self.C_attention_out = channel // C_reduction
        self.T_attention_out = time // T_reduction
        self.S_Convs = nn.ModuleList([])
        for i in kernels:
            self.S_Convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', layer.Conv2d(channel, channel, kernel_size=i, padding=i // 2)),
                    ('bn', layer.BatchNorm2d(channel)),
                    ('lif', neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()))
                ]))
            )

        # --------------------------------channel---------------------------------
        self.C_before_attention_conv = layer.Conv2d(channel, self.C_attention_out, 1)
        self.C_after_attention_convs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.C_after_attention_convs.append(layer.Conv2d(self.C_attention_out, channel, 1))

        # --------------------------------time---------------------------------
        self.T_before_attention_conv = nn.Conv3d(time, self.T_attention_out, 1)
        self.T_after_attention_convs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.T_after_attention_convs.append(nn.Conv3d(self.T_attention_out, time, 1))

        # ----------------------------------------------------------------------
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        T, B, C, _, _ = x.size()
        S_Conv_outs = []
        for conv in self.S_Convs:
            S_Conv_outs.append(conv(x))
        multiscale_feature_map = torch.stack(S_Conv_outs, 0)    # [n, T, B, C, h, w]
        feature_map = sum(S_Conv_outs)
        feature_map = feature_map.mean(-1).mean(-1)
        feature_map = feature_map.unsqueeze(-1).unsqueeze(-1)       # [T, B, C, 1, 1]
        # --------------------------------channel---------------------------------
        C_attention_weight = []     # [n]->[T, B, C, 1, 1](inside)
        C_attention_mid_weight = self.C_before_attention_conv(feature_map)
        C_attention_mid_weight = self.relu(C_attention_mid_weight)
        for conv in self.C_after_attention_convs:
            weight = conv(C_attention_mid_weight)
            C_attention_weight.append(weight.view(T, B, C, 1, 1))
        # ---------------------------------
        C_attention_weight_stack = torch.stack(C_attention_weight, 0)   # [n, T, B, C, 1, 1]
        C_attention_weight_stack = self.softmax(C_attention_weight_stack)
        # --------------------------------time---------------------------------
        T_attention_weight = []
        feature_map_before_T_attention = feature_map.permute(1, 0, 2, 3, 4)     # [B, T, C, 1, 1]
        T_after_pool = self.avg_pool(feature_map_before_T_attention)            # [B, T, 1, 1, 1]
        T_attention_mid_weight = self.T_before_attention_conv(T_after_pool)
        T_attention_mid_weight = self.relu(T_attention_mid_weight)
        for conv in self.T_after_attention_convs:
            weight = conv(T_attention_mid_weight)
            T_attention_weight.append(weight)
        # # ---------------------------------
        T_attention_weight_stack = torch.stack(T_attention_weight, 0)       # [n, B, T, 1, 1, 1]
        T_attention_weight_stack = T_attention_weight_stack.permute(0, 2, 1, 3, 4, 5)       # [n, T, B, 1, 1, 1]
        T_attention_weight_stack = self.softmax(T_attention_weight_stack)
        # ----------------------------------------------------------------------
        # T:[n, T, B, 1, 1, 1]                 C:[n, T, B, C, 1, 1]
        # The following section is the AZO regularization method
        # ------------------------------------------------------------------------------------------------------
        if self.AZO == True:
            if train_flag == 1:
                n_C = C // self.RCR           # Take the minimum number of T
                n_T = T // self.RTR           # Take the minimum number of C


                T_min_indices = []              # [4, n_T]
                C_min_indices = []              # [4, n_C]

                for i in T_attention_weight_stack:      # [T, B, 1, 1, 1]
                    j = torch.sum(i, dim=1)             # Sum along the time dimension for easy finding of the minimum value
                    j = j.squeeze()                     # 16

                    T_min_indices.append(torch.topk(j, k=n_T, largest=False).indices)         # Indices of the smallest n numbers

                j = 0               # Indicates which scale is being processed
                for i in C_attention_weight_stack:      # [T, B, C, 1, 1]
                    c = torch.sum(i, dim=1)             # Sum along the batch dimension for easy finding of the minimum value    [T, C, 1, 1]
                    # Find n_C channels at the time steps specified by T_min_index
                    t = T_min_indices[j]                # [n_T]
                    select_channel = []                 # [n_T]->[B, C, 1, 1]
                    for m in range(len(t)):
                        select_channel.append(i[t[m]])
                    select_channel = torch.stack(select_channel, dim=0)         # [n_T, B, C, 1, 1]
                    k = torch.sum(select_channel, dim=0)                    # Sum along n_T dimension  [B, C, 1, 1]
                    k = torch.sum(k, dim=0)                                 # Sum along B         [C, 1, 1]
                    k = k.squeeze()                                         # [C]

                    C_min_indices.append(torch.topk(k, k=n_C, largest=False).indices)         # Indices of the smallest n numbers

                    j += 1
                # ------------------------------------------------------------------------------------------------------

                attention_module_out = (T_attention_weight_stack * C_attention_weight_stack
                                        * multiscale_feature_map)

                after_replace = attention_module_out       # [n, T, B, C, h, w]

                for i in range(len(T_min_indices)):             # Scales
                    for j in T_min_indices[i]:      # Time steps with the smallest attention weights
                        for k in C_min_indices[i]:  # Channels with the smallest attention weights
                            if j != 0:
                                after_replace[i, j, :, k] =\
                                    attention_module_out[i, j - 1, :, k]


                result = after_replace.sum(0)

            else:
                result = (T_attention_weight_stack * C_attention_weight_stack
                                        * multiscale_feature_map).sum(0)
        else:
            result = (T_attention_weight_stack * C_attention_weight_stack
                                        * multiscale_feature_map).sum(0)

        return result
