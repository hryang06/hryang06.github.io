---
title: STR module details
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

What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis에서 등장하는 modele을 정리한 글입니다.


## 1. Transformation Stage

*[[STN]](https://arxiv.org/abs/1506.02025) Spatial Transformer Network*

*[[RARE]](https://arxiv.org/abs/1603.03915) Robust Scene Text Recognition with Automatic Rectification*

---

**STN (Spatial Transformer Network)**
- CNN max pooling : feature map을 축소하여, 분류 작업에 유리한 invariance를 얻을 수 있음. 예를 들어, 얼굴 인식에서 위치와 상관없이 눈, 코, 입을 인식할 수 있도록 한다.
- BUT! translation(위치)와 관계없이 동일하게 인식하지만, orientation(방향), proportion(비율) 등 달라지면 인식을 못한다. (물체를 바라보는 시점도 취약)
- data augmentation(다양한 방식으로 변형하여 이미지 생성)을 사용하면 이를 극복할 수 있으나 학습 시간이 길어진다.
- 이러한 CNN의 spatially inavariant(불변성)를 한계라고 생각하여 STN 등장하였다. (max pooling의 2*2 단위 연산으로는 data의 다양한 spatial vaiability(scale, rotation, translation) 대처하기 힘들다.)
- STN은 image 또는 feature map에서 관련 영역만 선택하여 가장 일반적인 형태로 변환 등 활용이 가능하다. 또한 scaling, cropping, rotation, non-grid deformation(thin plate spline) 등 지원한다.

    ![Figure 2](/assets/images/post/stn/figure2.PNG)

---

### 1.1 TPS transformation

- input image $$X$$ -> normalized image $$\tilde X$$
- fiducial points set(F 개) 사이에서 smooth spline interpolation을 사용한다.

#### 1\) [**localization network**](#1.2-tps-implementation)
: finding a text boundary

input image $$X$$ 위에 존재하는 fiducial points의 x-y좌표 $$C$$를 계산한다. $$\tilde C$$는 초기화되는 좌표로 normalized image(rectified image)에서의 fiducial points를 의미한다.

$$C = [c_1, ... , c_F] \in \mathbb{R^{2*F}}, c_f = [x_f, y_f]^T$$<br>
$$\tilde C$$ : normalized image $$\tilde X$$의 pre-defined top & bottom location

![Figure 6](/assets/images/post/stn/figure6.PNG)

#### 2\) **grid generator** 
: linking the location of the pixels in the boundary to those of the normalized image

![Figure 3](/assets/images/post/stn/figure3.PNG){: .center}

localization network에서 찾은 identified region과 normalized image(rectified image)를 연결하는 T를 찾는다.

![Formula 1](/assets/images/post/str/formula1.PNG)  $$T \in \mathbb{R^{2*F+3}}$$

![Formula 2](/assets/images/post/str/formula2.PNG)  $$R = \{d_{ij}^2\}, d_{ij} = $$euclidean distance between $$\tilde c_i$$ & $$\tilde c_j$$

#### 3\) **image sampler** 
: generating a normalized image by using the values of pixels and the linking information

grid generator으로 결정된 input image의 픽셀을 interpolate하여 normalized image를 생성한다. 최종 output이 생성된다.

![Figure 4](/assets/images/post/stn/figure4.PNG){: .center}

### 1.2 TPS-Implementation

TPS는 input image의 fiducial points를 계산하는 localization network를 필요로 한다. RARE[[25]](#robust-scene-text-recognition-with-automatic-rectification)에서 사용한 요소에다가 Batch Normalization layers(BN)와 network의 training을 안정시키기 위해 adaptive average pooling(APool)을 추가한다.

![Table 4](/assets/images/post/str/table4.PNG)

- 4 convolution layer + batch normalization layer + 2x2 max-pooling layer(마지막은 adaptive average pooling)
- 모든 convolution layer는 filter size = 3, padding size = 1, stride size = 1
- APool 이후, two fully connected layers : 512 to 256, 256 to 2F
- final output : 2F dimesional vector (input image의 F fiducial points x-y 좌표와 일치)
- 모든 layer의 activation function = ReLU


## 2. Feature Extration Stage

input image $$X$$ or $$\tilde X$$ -> feature map $$V = \{v_i\}, ( i = 1, ... , I )$$ (num of columns in feature map)

### 2.1 VGG

CRNN[24], RARE[25]에서 사용한 VGG를 구현하였다.

![Table 5](/assets/images/post/str/table5.PNG)

-> output : 512 channels * 24 columns


### 2.2 RCNN (recurrently applied CNN)

*[[GRCNN]](https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf) Gated Recurrent Convolution Neural Network for OCR*

gating mechanism으로 recursive하게 적용할 수 있는 RCNN인 Gated RCNN(GRCNN)을 구현하였다.

![Table 6](/assets/images/post/str/table6.PNG)

-> output : 512 channels * 26 columns

- classification(물체 하나하나 인식)뿐만 아니라 object detection(bounding box로 다양한 object 인식)에서도 높은 성능을 보인다.


### 2.3 ResNet (Residual Network)

*[[ResNet]](https://arxiv.org/abs/1512.03385) Deep Residual Learning for Image Recognition*

- neural network의 구조가 deep 할수록 vanishing/exploding gradient 문제로 정확도가 줄어든다. (weight 분포가 균등하지 않고 역전파가 제대로 이뤄지지 않기 때문이다.) ResNet은 이러한 문제를 해결하기 위해 제안되었다.
- **residual learning**
    - 기존 네트워크 $$H(x)$$에서 $$F(x) = H(x) - x$$로 변형시켜 $$F(x) + x$$를학습시키는 것을 말한다.
    - shortcut connection : $$y = \mathcal{F}(x, \{W_i\} + x)$$
    ![Figure 2](/assets/images/post/resnet/figure2.PNG)<br>
    - ResNet은 deep하더라도 더 좋은 성능을 보였다.
    ![Figure 4](/assets/images/post/resnet/figure4.PNG)

- FAN[4]에서 사용한 것과 동일한 network를 사용하였다.
- 총 29 trainable layer가 있다.

![Table 7](/assets/images/post/str/table7.PNG)

-> output : 512 channels * 26 columns


## 3. Sequence Modeling Stage

$$H = Seq.(V)$$

### 3.1 BiLSTM (Bidirectional LSTM)

- CRNN[24]에서 사용한 2-layers BiLSTM을 구현하였다.
- FC layer를 포함한 모든 hidden state의 dimension은 256이다.
- Seq. module을 사용하지 않은 경우, H = V

---

**LSTM (Long Short Term Memory)**<br>
RNN(Recurrent Neural Networks)의 vanishing gradient problem을 극복하기 위해서 고안되었다. RNN의 hidden state에 cell state를 추가한 구조를 보인다.<br>
(참고 : <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>)

---


## 4. Prediction Stage

input $$H$$ -> final prediction $$Y = y_1,y_2,... $$ (sequence of characters)

C : character label set (37)


| pred. | examples |
|-------|----------|
| CTC | CRNN, GRCNN, Rosetta, STAR-Net |
| Attn | R2AM, RARE |

### 4.1 CTC (Connetionist Temporal Classification)

*[[CTC]](https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c) An Intuitive Explanation of Connectionist Temporal Classification*

- 음성 인식, OCR에서 많이 사용한다.
- character labe set C : 36 alphanumeric characters + 1 blank

$$Y \approx M(argmax\ p(\pi \vert H))$$

**1. Encoding**

특별한 문자 blank를 사용하여 중복 문자 문제를 해결한다. 예를 들어 to와 too라는 단어를 encoding하면 다음과 같다.

![Example](/assets/images/post/ctc/example.PNG)

too를 too로 encoding하면, 이후 decoding할 때 to라는 단어로 예측할 수 있다. 이러한 중복 문제를 해결하기 위해 blank를 사용한다. 여기서 '-t-o'와 'to', 그외에도 'too', 't-oo' 등은 모두 'to'를 가리키지만 이미지에서는 서로 다른 정렬을 보인다.

**2. Decoding**

![Figure 4](/assets/images/post/ctc/figure4.png)

best path decoding은 다음과 같다.
1. 매 time step마다 가장 높은 확률을 가지는 문자를 선택한다. (aaa-b)
2. 중복되는 문자를 먼저 제거하고, (a-b)
3. 모든 blank를 제거한다. (ab)

**3. Loss Function**

![Figure 3](/assets/images/post/ctc/figure3.png)

t는 time step이고, 세 가지 문자 {a, b, -}가 존재한다. 위의 그림을 따라 모든 경우에 대해 구할 수 있는데, 예를 들어 'aa'는 0.4*0.4 = 0.16이 나온다. 만약 ground truth 문자가 'a'라면, 'aa', 'a-', '-a'에 대해 모두 합하여 0.64라는 것을 알 수 있다. 여기서 0.64는 loss가 아니라 ground truth의 probability를 의미하므로, loss는 probability의 음의 로그를 취하면 된다.

---

**CRNN 구조 : None-VGG-BiLSTM-CTC**<br>
*[[CRNN]](https://arxiv.org/abs/1507.05717) An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition*<br>
1. 중복되는 문자를 제거한다. (-s-t-ate)
2. blank(-)를 제거한다. (state)

![CRNN_fig](/assets/images/post/ctc/crnn_fig.PNG){: .center}

---

### 4.2 Attn (Attention mechanism)

*[[Attn]](https://arxiv.org/abs/1709.02054) Focusing attention: Towards accurate text recognition in natural images*

*[[RARE]](https://arxiv.org/abs/1603.03915) Robust Scene Text Recognition with Automatic Rectification*

- FAN[4], AON[5], EP[2]에서 사용한 one layer LSTM attention decoder를 구현하였다.
- C : 36 alphanumeric characters + 1 EOS(end of sentence)

$$each\ step\ t, y_t = softmax(W_{_0S_t} + b_0)$$

**1. Encoding**

input을 sequential representation으로 변환

![Figure 5](/assets/images/post/attn/figure5.png)


**2. Decoding**

output sequence 생성



**3. Loss Function**



