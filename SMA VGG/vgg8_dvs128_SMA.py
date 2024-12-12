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
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from collections import OrderedDict

torch.manual_seed(3401)
np.random.seed(3401)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(3401)

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
                n_C = C // self.RCR           # Take the smallest few T's
                n_T = T // self.RTR           # Take the smallest few C's
            
            
                T_min_indices = []              # [4, n_T]
                C_min_indices = []              # [4, n_C]
            
                for i in T_attention_weight_stack:      # [T, B, 1, 1, 1]
                    j = torch.sum(i, dim=1)             # Accumulate over time to facilitate finding the minimum value
                    j = j.squeeze()                     
            
                    T_min_indices.append(torch.topk(j, k=n_T, largest=False).indices)         # The minimum number of n indexes
            
                j = 0               # Indicate to what scale the processing has reached
                for i in C_attention_weight_stack:      # [T, B, C, 1, 1]
                    c = torch.sum(i, dim=1)             # Accumulate on the batch dimension to facilitate finding the minimum value, [T, C, 1, 1]
                    # Find nC channels at the time step specified in the Tmin_index
                    t = T_min_indices[j]                # [n_T]
                    select_channel = []                 # [n_T]->[B, C, 1, 1]
                    for m in range(len(t)):
                        select_channel.append(i[t[m]])
                    select_channel = torch.stack(select_channel, dim=0)         # [n_T, B, C, 1, 1]
                    k = torch.sum(select_channel, dim=0)                    # Accumulate on the n-T dimension  [B, C, 1, 1]
                    k = torch.sum(k, dim=0)                                 # Accumulate on B    [C, 1, 1]
                    k = k.squeeze()                                         # [C]
            
                    C_min_indices.append(torch.topk(k, k=n_C, largest=False).indices)         # The minimum number of n indexes
            
                    j += 1
                # ------------------------------------------------------------------------------------------------------
            
                attention_module_out = (T_attention_weight_stack * C_attention_weight_stack
                                        * multiscale_feature_map)
            
                after_resplace = attention_module_out.clone()       # [n, T, B, C, h, w]
            
                for i in range(len(T_min_indices)):  # scale
                    non_zero_T_min_indices = torch.stack([j for j in T_min_indices[i] if j != 0])
                    non_zero_T_min_indices = non_zero_T_min_indices[:, None]
                    C_min_indices[i] = C_min_indices[i][None, :]
                    after_resplace[i, non_zero_T_min_indices, :, C_min_indices[i]] = attention_module_out[i,
                                                                                    non_zero_T_min_indices - 1, :,
                                                                                    C_min_indices[i]]
            
                result = after_resplace.sum(0)
            
            else:
                result = (T_attention_weight_stack * C_attention_weight_stack
                                        * multiscale_feature_map).sum(0)
        else:
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

parser = argparse.ArgumentParser(description='SMA-VGG DVS128 Gesture Training')

parser.add_argument('--device', default='cuda:0', help='Device for execution\n')
parser.add_argument('--log-dir', default='./runs', help='Location to save tensorboard log files.')
parser.add_argument('--model-output-dir', default='./result_model', help='Path to save models and results, e.g., "./"\n')
parser.add_argument('--num-workers', default=70, type=int, help='Number of CPU cores to use for loading datasets\n')
parser.add_argument('-b', '--batch-size', default=9, type=int, help='Batch size, e.g., "64"\n')
parser.add_argument('-T', '--timesteps', default=16, type=int, dest='T', help='Duration of simulation, e.g., "100"\n')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='Learning rate, e.g., "1e-3"\n', dest='lr')
parser.add_argument('-N', '--epoch', default=200, type=int, help='Training epochs\n')
parser.add_argument('--decay', default=200, type=int, help='Number of learning rate decay times\n')
parser.add_argument('--local-rank', default=-1, type=int, help='Multi-GPU training, indicates the number of GPUs (processes)')

train_flag = 1

def main():
    ''' Conduct one test per epoch '''
    args = parser.parse_args()
    print("############## Parameter details ##############")
    # print("########## Configurations ##########")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))  # Output all parameters
    print("###############################################")

    global train_flag

    filename = (os.path.basename(__file__))[0:-3]
    device = args.device
    log_dir = args.log_dir
    model_output_dir = args.model_output_dir
    num_workers = args.num_workers
    batch_size = args.batch_size
    num_steps = args.T
    lr = args.lr
    epochs = args.epoch
    decay = args.decay
    local_rank = args.local_rank            # Number of graphics cards

    # DDP：DDP initialization
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # NCCL is the fastest and most recommended backend on GPU devices

    writer = SummaryWriter(log_dir)  # Used for tensorboard

    max_test_accuracy = 0  # Record the highest testing accuracy
    loss_list = []
    train_accs = []
    test_accs = []  # Record testing accuracy
    spiking_rate = []
    test_loss_list = []

    scaler = amp.GradScaler()
    functional.set_backend(Net, 'cupy', instance=neuron.LIFNode)

    train_dataset = DVS128Gesture('../data/DVS128Gesture', train=True, data_type='frame', split_by='number',
                              frames_number=num_steps)
    test_dataset = DVS128Gesture('../data/DVS128Gesture', train=False, data_type='frame', split_by='number',
                             frames_number=num_steps)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)  # sampler 
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        sampler=test_sampler
    )

    print(f"Training dataset size: {len(train_dataset)}, Testing dataset size: {len(test_dataset)}")

    net = Net
    print(net)
    # Constructing models
    # DDP: The Load model needs to be loaded on the master before constructing the DDP model.
    net = net.to(local_rank)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)

    # Regularization of learning rate
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay)

    def cal_firing_rate(s_seq):
        return s_seq.flatten(0).mean(0)
    
    # Monitor for recording Spiking rate
    if dist.get_rank() == 0:
        spike_seq_monitor = monitor.OutputMonitor(net, neuron.LIFNode, cal_firing_rate)

    for epoch in range(1, epochs + 1):
        # DDP: Set the epoch of the sampler,
        # DistributedSampler requires this to specify the shuffle method,
        # By maintaining the same random number seed between different processes, different processes can achieve the same shuffle effect.
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)

        train_correct_sum_in_epoch = 0        # The number of correctly predicted data in each epoch used for training
        train_data_sum_in_epoch = 0           # The total amount of data used for training in each epoch
        print(f"train epoch : {epoch}")
        net.train()
        train_flag = 1
        if dist.get_rank() == 0:
            spike_seq_monitor.clear_recorded_data()     # Clear recorded Spiking rates
            spiking_rate.clear()                        # Clear the list of spike rates in the record
            spike_seq_monitor.disable()         # Stop recording during training phase
        for img, label in tqdm(train_loader):                       # Number of cycles ≈ Total data/batch size
            img = img.to(local_rank)
            img = img.transpose(0, 1)
            label = label.to(local_rank)
            label_one_hot = F.one_hot(label, 11).float()            # Encode the labels one hot for later loss calculation
            # Mixed precision training
            with amp.autocast():
                result = net(img).mean(0)
                # Calculate the number of correctly classified items
                pred = result.argmax(dim=1)
                correct = pred.eq(label).sum().float().item()  # Correct quantity (float)
                train_correct_sum_in_epoch += correct  # Record the correct quantity for each epoch
                loss = F.mse_loss(result, label_one_hot)  # MSE loss
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            functional.reset_net(net)  # Reset network status
            train_data_sum_in_epoch += label.numel()  # The numel function returns the number of elements in an array, with a value of the total number of images
        # Accurate count of correct and total quantities across threads
        train_correct_sum_in_epoch = torch.tensor(train_correct_sum_in_epoch).to(local_rank)
        train_data_sum_in_epoch = torch.tensor(train_data_sum_in_epoch).to(local_rank)
        torch.distributed.all_reduce(train_correct_sum_in_epoch, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(train_data_sum_in_epoch, op=torch.distributed.ReduceOp.SUM)
        if dist.get_rank() == 0:
            loss_list.append(loss.item())
            train_accuracy_in_epoch = train_correct_sum_in_epoch / train_data_sum_in_epoch  # Accuracy for each epoch
            train_accs.append(train_accuracy_in_epoch)
            writer.add_scalars('train_and_test_accuracy', {'train': train_accuracy_in_epoch}, epoch)  # Storing accuracy for each epoch in tensorboard
            writer.add_scalar('loss_epoch', loss.item(), epoch)  # Storing loss
        lr_scheduler.step()  # Executing learning rate regularization


        print("###############   End of training, start testing   ###############")
        net.eval()
        train_flag = 0
        if dist.get_rank() == 0:
            spike_seq_monitor.enable()  # Start recording Spiking rate
        with torch.no_grad():  # Perform testing for each epoch
            test_data_correct_sum = 0  # Number of correct model outputs during testing
            test_data_sum = 0  # Total data during testing
            for img, label in tqdm(test_loader):
                img = img.to(local_rank)
                img = img.transpose(0, 1)
                label = label.to(local_rank)
                result = net(img).mean(0)
                label_one_hot_test = F.one_hot(label, 11).float()  # One-hot encode labels for calculating loss
                loss_test = F.mse_loss(result, label_one_hot_test)
                # Calculate the number of correct classifications
                pred = result.argmax(dim=1)
                correct = pred.eq(label).sum().float().item()
                test_data_correct_sum += correct  # Accumulate correct count
                test_data_sum += label.numel()  # Accumulate total data count
                functional.reset_net(net)
            # Accurate count of correct and total quantities across threads
            test_data_correct_sum = torch.tensor(test_data_correct_sum).to(local_rank)
            test_data_sum = torch.tensor(test_data_sum).to(local_rank)
            torch.distributed.all_reduce(test_data_correct_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(test_data_sum, op=torch.distributed.ReduceOp.SUM)
        if dist.get_rank() == 0:
            test_accuracy = test_data_correct_sum / test_data_sum  # Testing accuracy
            writer.add_scalars('train_and_test_accuracy', {'test': test_accuracy}, epoch)  # Storing test accuracy
            test_accs.append(test_accuracy)  # Storing test accuracy
            test_loss_list.append((loss_test))
            # Determine if this result is the best so far
            save_max = False
            if test_accuracy > max_test_accuracy:
                max_test_accuracy = test_accuracy  # Store the highest accuracy
                save_max = True  # Indicate the current best model to be output later
            print(f"epoch{epoch}, train accuracy: {train_accuracy_in_epoch}, test accuracy: {test_accuracy}, max test accuracy so far: {max_test_accuracy}")
            writer.close()  # Close tensorboard write

            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
            if not os.path.exists(model_output_dir + '/spiking_rate/' + filename):
                os.makedirs(model_output_dir + '/spiking_rate/' + filename)

            # Cannot save the model concurrently with the hook created to save spiking rate
            # if save_max:
            #     torch.save(net.state_dict(), os.path.join(model_output_dir, filename + '_max.pt'))
            
            # Save position list only in the last epoch
            if epoch == epochs:
                torch.save(spike_seq_monitor.monitored_layers, model_output_dir + '/spiking_rate/' +
                        filename + '/' + 'name_list.pt')
            
            # Format firing rate
            for i in range(len(spike_seq_monitor.monitored_layers)):
                spiking_rate.append(spike_seq_monitor[spike_seq_monitor.monitored_layers[i]][0])

            if save_max == True:
                # Save spiking rate
                torch.save(spiking_rate, model_output_dir + '/spiking_rate/' +
                            filename + '/' + 'spikingrate_' + 'epoch' + str(epoch) + '.pt')




    if dist.get_rank() == 0:
        train_accs = np.array(torch.tensor(train_accs, device='cpu'))
        np.save(model_output_dir + '/' + filename + '_train_acc.npy', train_accs)  # Store the training accuracy list for each epoch in binary file format
        test_accs = np.array(torch.tensor(test_accs, device='cpu'))
        np.save(model_output_dir + '/' + filename + '_test_acc.npy', test_accs)  # Store the test accuracy list for each epoch in binary file format
        loss_list = np.array(torch.tensor(loss_list, device='cpu'))
        np.save(model_output_dir + '/' + filename + '_loss.npy', loss_list)  # Store the loss list for each epoch in binary file format
        test_loss_list = np.array(torch.tensor(test_loss_list, device='cpu'))
        np.save(model_output_dir + '/' + filename + '_test_loss.npy', test_loss_list)  # Store the test loss list for each epoch in binary file format



if __name__ == '__main__':
    main()


