---
title: STR framework details (appendix D)
layout: single
author_profile: true
read_time: true
use_math: true
comments: true
share: true
related: true
categories:
- str
toc: true
toc_sticky: true
toc_label: 목차
article_tag1: STR
article_tag2: scene text recognition
last_modified_at: 2020-04-09 11:38:00 +0800
---

What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis의 Appendix D를 정리한 글입니다.


## 1. Transformation Stage

![Figure 11](/assets/images/post/str/figure11.PNG)

TPS를 추가하는 것과 하지 않는 것은 크게 차이가 없다. 하지만, TPS를 사용하는 것이 사용하지 않을 때보다 시간이 길어지고 메모리가 약간 더 사용하지만 정확도가 높아진다.

### TPS transformation

input image $$X$$ -> normalized image $$\tilde X$$

fiducial points set(F 개) 사이에서 smooth spline interpolation을 사용한다.

1\) **localization network** : finding a text boundary(#tps-implementation)

input image $$X$$ 위에 존재하는 fiducial points의 x-y좌표 $$C$$를 계산한다.

$$C = [c_1, ... , c_F] \in \mathbb{R^2*F}, c_f = [x_f, y_f]^T$$

$$\tilde C$$는 normalized image $$\tilde X$$의 pre-defined 위아래 위치이다.

2\) **grid generator** : linking the location of the pixels in the boundary to those of the normalized image

mapping function (localization network에서 한 정한 identified region - normalized images)

![Formula 1](/assets/images/post/str/formula1.PNG)  $$T \in \mathbb{R^2*(F+3)}$$

![Formula 2](/assets/images/post/str/formula2.PNG)  $$R = \{d_ij^2\}, d_ij = $$euclidean distance between $$\tilde c_i$$ - $$\tilde c_j$$

3\) **image sampler** : generating a normalized image by using the values of pixels and the linking information

grid generator으로 결정된 input image의 픽셀을 interpolate하여 normalized image를 생성한다.


### TPS-Implementation

TPS는 input image의 fiducial points를 계산하는 localization network를 필요로 한다. RARE[25]에서 사용한 요소에다가 Batch Normalization layers(BN)와 network의 training을 안정시키기 위해 adaptive average pooling(APool)을 추가한다.

![Table 4](/assets/images/post/str/table4.PNG)

- 4 convolution layer + batch normalization layer + 2x2 max-pooling layer(마지막은 adaptive average pooling)
- 모든 convolution layer는 filter size = 3, padding size = 1, stride size = 1
- APool 이후, two fully connected layers : 512 to 256, 256 to 2F
- final output : 2F dimesional vector (input image의 F fiducial points x-y 좌표와 일치)
- 모든 layer의 activation function = ReLU


## 2. Feature Extration Stage

input image $$X$$ / $$\tilde X$$ -> feature map $$V = {v_i}, i = 1, ... , I$$(num of columns in feature map)

![Figure 12](/assets/images/post/str/figure12.PNG)

VGG가 가장 시간이 적게 걸리지만 낮은 정확도를 보인다. RCNN은 가장 시간이 걸리지만 가장 적은 메모리와 VGG보다 높은 정확도를 보인다. ResNet이 가장 높은 정확도를 가지지만, 다른 module에 비해 많은 메모리를 필요로 한다. 따라서, 메모리 제약이 존재하면 RCNN이 가장 좋은 trade-off이고, 그렇지 않다면 ResNet을 사용해야 한다. 시간적인 측면에서 세 module 모두 차이가 크지 않으므로 극단적인 경우에만 고려한다.


### 1) VGG

CRNN[24], RARE[25]에서 사용한 VGG를 구현하였다.

![Table 5](/assets/images/post/str/table5.PNG)

-> output : 512 channels * 24 columns


### 2) RCNN (recurrently applied CNN)

gating mechanism으로 recursive하게 적용할 수 있는 RCNN인 Gated RCNN(GRCNN)을 구현하였다.

![Table 6](/assets/images/post/str/table6.PNG)

-> output : 512 channels * 26 columns


### 3) ResNet (Residual Network)

FAN[4]에서 사용한 것과 동일한 network를 사용하였다. 총 29 trainable layer가 있다.

![Table 7](/assets/images/post/str/table7.PNG)

-> output : 512 channels * 26 columns



## 3. Sequence Modeling Stage

$$H = Seq.(V)$$

![Figure 13](/assets/images/post/str/figure13.PNG)

TPS와 비슷하지만, BiLSTM을 사용하면 비슷한 시간과 메모리에 비해 더 높은 정확도로 향상시킨다.

### 1) BiLSTM (Bidirectional LSTM)

CRNN[24]에서 사용한 2-layers BiLSTM을 구현하였다.

FC layer를 포함한 모든 hidden state의 dimension은 256이다.

\* Seq. module을 사용하지 않은 경우, H = V



## 4. Prediction Stage

input $$H$$ -> final prediction $$Y = y_1,y_2,... $$ (sequence of characters)

C : character labe set (37)

![Figure 14](/assets/images/post/str/figure14.PNG)

Attn은 CTC에 비해 정확도를 높이려고 할 때 시간이 오래 걸린다.

### 1) CTC (Connetionist Temporal Classification)

C : 36 alphanumeric characters + 1 blank

mapping function M : 반복되는 문자와 blank를 제거함으로써 map한다.

$$Y \approx M(argmax p(\pi \vert H))$$


### 2) Attn (Attention mechanism)

FAN[4], AON[5], EP[2]에서 사용한 one layer LSTM attention decoder를 구현하였다.

C : 36 alphanumeric characters + 1 EOS(end of sentence)

$$each time step t, y_t = softmax(W_0_S_t + b_0)$$
