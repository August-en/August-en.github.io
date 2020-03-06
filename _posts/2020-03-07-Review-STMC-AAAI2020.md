---
layout:     post   				    # 使用的布局（不需要改）
title:      Review STMC AAAI2020 				# 标题 
subtitle:   Hello World, Hello Blog #副标题
date:       2020-03-07 				# 时间
author:     Wang						# 作者
header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - Sign Language
---
## Spatial-Temporal Multi-Cue Network for Continuous Sign Language Recognition

来自中科大的文章，发表在AAAI2020上。

提出了一种利用空间和时间上的多种信息相结合的端到端学习框架，RWTH-v2 WER: **21.1**，RWTH-v3 WER: **19.6**，达到了最新的State-Of-The-Art。

主要创新点在于：1.空间建模。利用姿态估计预测人体7对关键点，再基于关键点为中心，剪裁出与手语强相关的左右手的patch，人脸patch等的信息，作为多种信息来源进行特征提取，最后输出人脸，人手，全图，姿态4种来源的特征向量。2.时间建模。时间建模有两条路径，<u>一条是多信号间的时间建模，另一条是多信号内的时间建模</u>，充分挖掘了各信号的时间信息。

此外沿用了前些年文章里提出的staged optimization strategy生成伪标签进行迭代优化。

### Outlines

1. STMC Architecture
2. Spatial Multi-Cue Module
3. Temporal Multi-Cue Module
4. Loss function and Inference
5. Details of the experiments
6. Results

### 1. STMC Architecture

![2020-02-23_18:05:01-DQdCE5-6xiz4V](https://raw.githubusercontent.com/August-en/image_hosting_service/master/images/2020-02-23_18:05:01-DQdCE5-6xiz4V.png)

- 网络的最终输出分为两部分：Inter-cue path与Intra-cue path，这两部分的输出与姿态估计的输出加权在一起计算Loss。最终的Inference过程仅由Inter-cue path输出预测值。（否则计算量高，速度慢？）
- 图中N指的是信息来源的个数，文中N=4，分别来源于full-frame, hands, face, pose。

### 2. Spatial Multi-Cue Module

<img src="https://raw.githubusercontent.com/August-en/image_hosting_service/master/images/2020-02-23_18:22:30-64g7ve-2ACJpm.png" alt="2020-02-23_18:22:30-64g7ve-2ACJpm" style="zoom:50%;" />

- SMC模块中，作者在VGG11 backbone的基础上，新设计了一个独立的Pose Estimation支路，预测7个关键点，分别是鼻子，左右肩，左右肘，左右手腕。增添的独立的Pose Estimation支路起到了正则化的作用，缓解了网络的过拟合程度。
- 得到关键点后，1.分别以关键点为中心，按照**固定大小**裁剪出人脸，左右手的图像块，用于生成Multi-cues进而进行特征提取。2. 得到人体姿态特征向量。（细节：裁剪时注意不要越界，HRNet用于生成keypoints annotations）
- 获取Spatial Multi-Cue Representation vector。分别来自full-frame, left and right hands, face, pose，维度如图所示，注意左右手的特征提取时采用了参数共享的卷积。

### 3. Temporal Multi-Cue Module

<img src="https://raw.githubusercontent.com/August-en/image_hosting_service/master/images/2020-02-23_21:02:18-xN82u4-VjYwcv.png" alt="2020-02-23_21:02:18-xN82u4-VjYwcv" style="zoom:50%;" />

作者提出的TMC Module旨在从inter-cue和intra-cue两个方面整合时空信息，而不是简单的信息融合。

> The intra-cue path captures the unique features of each visual cue.
>
> The inter-cue path learns the combination of fused features from different cues at different time scales.

#### 3.1 Intra-Cue Path（信号内）

> The first path is to provide unique fea- tures of different cues at different time scales. 

<img src="https://raw.githubusercontent.com/August-en/image_hosting_service/master/images/2020-02-23_21:10:23-f224VZ-D9ngj5.png" alt="2020-02-23_21:10:23-f224VZ-D9ngj5" style="zoom:50%;" />

- k = 5, N = 4, C = 1024，$K^\frac{C}{N}_k$为时间卷积核（即一维卷积核）
- 该路径分别对4种信号的vector进行kernel_size = 5的 conv_relu 运算，再将4种信号的vector concate为1个vector，变量的维度如公式 (5) (6)中所示。

#### 3.2 Inter-Cue Path（信号间）

> The second path is to perform the temporal transformation on the inter-cue feature from the previous block and fuse information from the intra-cue path as follows.

<img src="https://raw.githubusercontent.com/August-en/image_hosting_service/master/images/2020-02-23_22:20:40-ujAX9U-clISC6.png" style="zoom:50%;" />

- $K^\frac{C}{2}_1$实现了维度变换(1024 -> 512)
- 该路径实现了对前一Inter-cue vector的时间变换及对该模块中Intra-cue vector的融合

在每个Block之后，有TP为kernel size = 2, stride = 2的Temporal max-pooling运算。

### 4. Loss function and Inference

#### 4.1 Loss function

在训练过程中，作者将Inter-cue path作为主要优化目标。为了提供每个单独信息特征的融合，Intra-cue path作起到辅助作用。因此，整个STMC框架的目标函数如下：

<img src="https://raw.githubusercontent.com/August-en/image_hosting_service/master/images/2020-02-23_22:42:31-fVR1a4-ZQwzHe.png" alt="2020-02-23_22:42:31-fVR1a4-ZQwzHe" style="zoom:50%;" />

- $\alpha$用于控制辅助loss的比重，$\beta$用于使姿态估计回归损失与其他损失处于相同的数量级
- $L^\beta_R$为smooth-L1 loss用于姿态估计的目标函数

<img src="https://raw.githubusercontent.com/August-en/image_hosting_service/master/images/2020-02-23_22:48:14-vbtoJn-FIaECZ.png" alt="2020-02-23_22:48:14-vbtoJn-FIaECZ" style="zoom:50%;" />

#### 4.2 Inference

> For inference, we pass video frames through the SMC and TMC modules. **Only the <u>inter-cue</u> feature sequence and its BLSTM encoder are used to generate the pos- terior probability distribution of glosses at all time steps.** We use the **beam search** decoder (Hannun et al. 2014) to search the most probable sequence within an acceptable range.(the beam width is set to 20)

### 5. Details of the experiments

- 为了获得关键点位置用于训练，作用使用了开源的HRNet工具去估计文中所述上半身7个关键点。

- Input frames are resized to 224 x 224
- Random crop at the same location of all frames, random discard of 20% frames, random flip all fr ames
- Inter-cue features, output channels after TCOVs and BLSTM are all set to 1024
- Intra-cue features, output channels atfer TCOVs and BLSTM are all set to 256
- Adam, lr = 5e-5, batch_siz e= 2, $\alpha$ = 0.6, $\beta$ = 30

Staged optimization strategy:

> First, we train a VGG11-based network as DNF (Cui, Liu, and Zhang 2019) and use it to decode pseudo labels for each clip. Then, we add a fully-connected layer after each output of the TMC module. The STMC network without BLSTM is trained with cross-entropy and smooth-L1 loss by SGD optimizer. The batch size is 24 and the clip size is 16. Finally, with fine- tuned parameters from the previous stage, our full STMC network is trained end-to-end under joint loss optimization.

### 6. Results

<img src="https://raw.githubusercontent.com/August-en/image_hosting_service/master/images/2020-02-23_23:15:51-HRxsMF-sE41vW.png" alt="2020-02-23_23:15:51-HRxsMF-sE41vW" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/August-en/image_hosting_service/master/images/2020-02-23_23:16:10-QSumQr-JL9p5H.png" alt="2020-02-23_23:16:10-QSumQr-JL9p5H" style="zoom:50%;" />

![2020-02-23_23:17:06-K4cuzH-wsV89Z](https://raw.githubusercontent.com/August-en/image_hosting_service/master/images/2020-02-23_23:17:06-K4cuzH-wsV89Z.png)

<img src="https://raw.githubusercontent.com/August-en/image_hosting_service/master/images/2020-02-23_23:17:36-yaIkSA-gQnOPE.png" alt="2020-02-23_23:17:36-yaIkSA-gQnOPE" style="zoom:50%;" />
