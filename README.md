# Open Source for Spiking-Multiscale-Attention
We have provided training programs and pre trained weights for the DVS128 Gesture, CIFAR10-DVS and N-Caltech101 datasets.
The appendix of the article can be accessed here: [link](https://drive.google.com/file/d/1cnr6Qay6-vL8wcjPesJY_JpzM3kNP6pe/view?usp=drive_link)

## Pre training weights(one GPU can load and run.)
|        **model**         | **datasets** | **Models** |
| :----------------------: | :--------: | :--------: |
| SMA-VGG | Dvs128 Gesture |  [link](https://drive.google.com/file/d/1IpiNyIzGoFZPB6VadyVDC4NwIjt3KEgX/view?usp=drive_link)    |
| SMA-AZO-VGG |Dvs128 Gesture |  [link](https://drive.google.com/file/d/1YhRPXQQlWmXr5404j_RjiHtvlarHWyPx/view?usp=drive_link)    |
| SMA-VGG | CIFAR10-DVS  | [link]()    |
| SMA-AZO-VGG |CIFAR10-DVS |  [link](https://drive.google.com/file/d/1zXzSUedGMBkqiCIQNKvLPQxoCaDi-uGQ/view?usp=drive_link)    |
| SMA-VGG | N-Caltech101 | [link](https://drive.google.com/file/d/1IBwLOPAtwJBbB7jwg9dB0TdJbMgFoD49/view?usp=drive_link)    |
| SMA-AZO-VGG | N-Caltech101 | [link](https://drive.google.com/file/d/1-YdptrxyELPF17qpRNVA54miwnsVZTGX/view?usp=drive_link)    |

PS: I have conducted an additional verification and can hereby confirm that all connections are now directly accessible without the need for prior authorization. Should you encounter any difficulties, please do not hesitate to contact me promptly. I extend my sincere apologies to the three colleagues who previously sought to obtain the weights and were required to submit formal applications to me.

## Operating environment
As described in the appendix of the paper, we utilized three devices in our experiments. Device one was dedicated to conducting experiments on the DVS128 Gesture, CIFAR10-DVS, and N-Caltech101 datasets. Device two was allocated for experiments involving the Imagenet-1K dataset using the ResNet18/34 architecture. Lastly, Device three was employed for experiments on the Imagenet-1K dataset using the ResNet104 network.

The specific configurations of these three devices are shown in the table below:
![image](https://github.com/Ym-Shan/Spiking_Multiscale_Attention_Arxiv/assets/121172737/e40cbb63-4d3d-4aac-8e6d-2d9381f44539)


Regardless of the configuration, the only core libraries used are [spikingjelly==0.0.0.0.14](https://github.com/fangwei123456/spikingjelly), [einops](https://github.com/arogozhnikov/einops), timm and [cupy](https://github.com/cupy/cupy).

For other unimportant configurations, please refer to `requirements.txt`

## Run the DVS128 Gesture
```
CUDA_VISIBLE_DEVICE="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 vgg8_dvs128_SMA.py
```

## Run the CIFAR10-DVS
```
CUDA_VISIBLE_DEVICE="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 vgg8_cifar10dvs_SMA.py
```

## Run the N-Caltech101
```
CUDA_VISIBLE_DEVICE="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 vgg8_NCaltech101_SMA.py
```

## The dataset visualization methods used in this paper have been integrated into the [SpikingJelly](https://github.com/fangwei123456/spikingjelly) framework：
[Update](https://github.com/fangwei123456/spikingjelly/pull/541)
```
A method save_as_pic has been added to save each frame of an individual event as a .png file. Prior to this, spikingjelly only had the method play_frame to save event data as .gif format.

A method save_every_frame_of_an_entire_DVS_dataset has been added that requires only one line of code to save each frame of every sample in an entire DVS dataset as a .png file.
```

## Please Citing
```
@inproceedings{shan2025advancing,
  title={Advancing spiking neural networks towards multiscale spatiotemporal interaction learning},
  author={Shan, Yimeng and Zhang, Malu and Zhu, Rui-jie and Qiu, Xuerui and Eshraghian, Jason K and Qu, Haicheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={2},
  pages={1501--1509},
  year={2025}
}
```
