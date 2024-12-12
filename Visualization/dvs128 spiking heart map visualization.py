import sys
import pickle
import torch
import torch.nn as nn
import torchvision
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
sys.path.append("../../..")
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, model, layer, monitor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda import amp
import os
import random
import time
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import matplotlib.colors as mcolors
import seaborn as sns


class SMA(nn.Module):
    def __init__(self, channel=512, time=16, kernels=[1, 3, 5, 7], C_reduction=16, T_reduction=4):
        super().__init__()
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
        self.T_before_attention_conv = nn.Conv3d(time, T_reduction, 1)
        self.T_after_attention_convs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.T_after_attention_convs.append(nn.Conv3d(T_reduction, time, 1))

        # ----------------------------------------------------------------------
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        T, B, C, _, _ = x.size()
        S_Conv_outs = []
        for conv in self.S_Convs:
            S_Conv_outs.append(conv(x))
        multiscale_feature_map = torch.stack(S_Conv_outs, 0)  # [n, T, B, C, h, w]
        feature_map = sum(S_Conv_outs)
        feature_map = feature_map.mean(-1).mean(-1)
        feature_map = feature_map.unsqueeze(-1).unsqueeze(-1)  # [T, B, C, 1, 1]
        # --------------------------------channel---------------------------------
        C_attention_weight = []  # [n]->[T, B, C, 1, 1](inside)
        C_attention_mid_weight = self.C_before_attention_conv(feature_map)
        C_attention_mid_weight = self.relu(C_attention_mid_weight)
        for conv in self.C_after_attention_convs:
            weight = conv(C_attention_mid_weight)
            C_attention_weight.append(weight.view(T, B, C, 1, 1))
        # ---------------------------------
        C_attention_weight_stack = torch.stack(C_attention_weight, 0)  # [n, T, B, C, 1, 1]
        C_attention_weight_stack = self.softmax(C_attention_weight_stack)
        # --------------------------------time---------------------------------
        T_attention_weight = []
        feature_map_before_T_attention = feature_map.permute(1, 0, 2, 3, 4)  # [B, T, C, 1, 1]
        T_after_pool = self.avg_pool(feature_map_before_T_attention)  # [B, T, 1, 1, 1]
        T_attention_mid_weight = self.T_before_attention_conv(T_after_pool)
        T_attention_mid_weight = self.relu(T_attention_mid_weight)
        for conv in self.T_after_attention_convs:
            weight = conv(T_attention_mid_weight)
            T_attention_weight.append(weight)
        # # ---------------------------------
        T_attention_weight_stack = torch.stack(T_attention_weight, 0)  # [n, B, T, 1, 1, 1]
        T_attention_weight_stack = T_attention_weight_stack.permute(0, 2, 1, 3, 4, 5)  # [n, T, B, 1, 1, 1]
        T_attention_weight_stack = self.softmax(T_attention_weight_stack)
        # ----------------------------------------------------------------------
        result = (T_attention_weight_stack * C_attention_weight_stack
                  * multiscale_feature_map).sum(0)

        return result


class VGG8(nn.Module):
    def __init__(self):
        super(VGG8, self).__init__()

        self.block1 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            ),

            layer.SeqToANNContainer(
                nn.BatchNorm2d(64),
            ),

            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),

            layer.SeqToANNContainer(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )
        )

        self.block2 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ),

            SMA(channel=128, C_reduction=4, time=16, T_reduction=4, AZO=True, RCR=10, RTR=4),

            layer.SeqToANNContainer(
                nn.BatchNorm2d(128),
            ),

            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),

            layer.SeqToANNContainer(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )
        )

        self.block3 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            ),

            SMA(channel=256, C_reduction=4, time=16, T_reduction=4, AZO=True, RCR=10, RTR=4),

            layer.SeqToANNContainer(
                nn.BatchNorm2d(256),
            ),

            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),

            layer.SeqToANNContainer(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )
        )

        self.block4 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            ),

            layer.SeqToANNContainer(
                nn.BatchNorm2d(512),
            ),

            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),

            layer.SeqToANNContainer(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )
        )

        self.block5 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ),

            layer.SeqToANNContainer(
                nn.BatchNorm2d(512),
            ),

            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),

            layer.SeqToANNContainer(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )
        )

        self.others = nn.Sequential(
            layer.AdaptiveAvgPool2d(output_size=3),
            layer.Flatten(start_dim=1, end_dim=-1),

            layer.Linear(in_features=4608, out_features=2048, bias=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),
            layer.Dropout(p=0.5),
            layer.Linear(in_features=2048, out_features=1024, bias=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),
            layer.Dropout(p=0.5),
            layer.Linear(in_features=1024, out_features=11, bias=True),
        )

    def forward(self, x):
        result1 = self.block1(x)
        result2 = self.block2(result1)
        result3 = self.block3(result2)
        result4 = self.block4(result3)
        result5 = self.block5(result4)
        result = self.others(result5)

        return result


Net = VGG8()

functional.set_step_mode(Net, step_mode='m')
Net.cuda()

parser = argparse.ArgumentParser(description='Visualization')

parser.add_argument('--device', default='cuda:0', help='Device to run on\n')
parser.add_argument('--log-dir', default='./runs', help='Location to save tensorboard log files.')
parser.add_argument('--num-workers', default=4, type=int, help='Number of cores used to load the dataset\n')
parser.add_argument('-b', '--batch-size', default=1, type=int, help='Batch size, e.g., "64"\n')
parser.add_argument('-T', '--timesteps', default=16, type=int, dest='T', help='Simulation duration, e.g., "100"\n')


def main():
    ''' Perform testing once per epoch '''
    args = parser.parse_args()
    device = args.device
    batch_size = args.batch_size
    num_steps = args.T

    test_set = DVS128Gesture('../../datasets/DVS128Gesture', train=False, data_type='frame', split_by='number', frames_number=num_steps)

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True)

    net = Net

    # Load weights
    checkpoint = torch.load('../weights/dvs128_AZO.pt')

    # Create new state_dict, correcting key names
    new_state_dict = {}
    for k, v in checkpoint.items():
        name = k.replace('module.', '')  # Remove "module." prefix
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)

    def cal_firing_rate(s_seq):
        # return s_seq.flatten(0).mean(0)
        return s_seq

    # Monitor for recording Spiking rate
    spike_seq_monitor = monitor.OutputMonitor(net, neuron.LIFNode, cal_firing_rate)

    net.eval()
    with torch.no_grad():              # Perform testing once per epoch
        i = 0
        toumingdu = 1
        for img, label in tqdm(test_loader):
            if i >= 0:
                img = img.float().to(device)
                img = img.transpose(0, 1)

                spike_seq_monitor.enable()  # Start recording Spiking rate

                net(img)

                for j in range(len(spike_seq_monitor.monitored_layers)):
                    if j == 5:        # Which Neuron.
                        for t in range(16):
                            pic = spike_seq_monitor[spike_seq_monitor.monitored_layers[j]][0].sum(dim=2).squeeze(1)[t]  # [T, H, W]
                            # Normalize tensor values to [0, 1] range for representing color intensity
                            normalized_tensor = (pic - pic.min()) / (pic.max() - pic.min())
                            colormap = cm.get_cmap('jet')
                            # Map normalized tensor values to colors
                            colored_image = colormap(normalized_tensor.cpu().numpy())
                            # Adjust transparency
                            colored_image[:, :, 3] = colored_image[:, :, 3] * toumingdu
                            # Save images using matplotlib
                            plt.imsave('./demo/' + str(i) + '_' + str(t) + '.png', colored_image)

            spike_seq_monitor.clear_recorded_data()
            functional.reset_net(net)
            i += 1



if __name__ == '__main__':
    main()
