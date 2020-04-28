---
title: STR analysis
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
article_tag3: OCR
article_tag4: optical character recognition
last_modified_at: 2020-04-24 15:26:00 +0800
---

What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis을 재연하고 결과를 정리한 글입니다. (appendix D 참고)

## 0. Setting Parameters

module
- Transformation: None / TPS
- FeatureExtraction: VGG / RCNN / ResNet
- SequenceModeling: None / BiLSTM
- Prediction: CTC / Attn

data
- train_data = data_lmdb_release/training
- valid_data = data_lmdb_release/validation
- batch_size = 192
- select_data = ['MJ', 'ST']
- batch_ratio = ['0.5', '0.5']
- total_data_usage_ratio = 1.0
- batch_max_length = 25 : maximum label length
- data_filtering_off = False

image
- imgH = 32
- imgW = 100
- rgb = False

characters
- character = 0123456789abcdefghijklmnopqrstuvwxyz : character label
- sensitive = False : 대소문자 구분 유무

training
- num_iter = 300000 : training iteration (epoch이라고 보면 될 듯)
- valInterval = 2000 : 몇번마다 validate할지 (validation마다 best accuracy와 norm ED를 찾아 saved_model에 저장한다.)
- saved_model = .pth
- FT = False
- adam = False : default인 Adadelta가 아닌 adam을 사용할지
- lr = 1 : learning rate [Adadelta default]
- beta1 = 0.9
- rho = 0.95 : decay rate rho [Adadelta default]
- eps = 1e-08 : 
- grad_clip = 5 : gradient clipping value

- PAD = False

TPS
- num_fiducial = 20 : fiducial point 개수

Feuture Extrator
- input_channel = 1
- output_channel = 512

LSTM
- hidden_size = 256 : hidden state size

Prediction (CTC/Attn)
- num_class = 37 / 38 : character label set [CTC = 37, Attn = 38]

etc
- manualSeed = 1111 : random seed setting
- workers = 4 : data loading workers 수
- num_gpu = 1

result
![Table 8](/assets/images/post/str/table8.PNG)

analysis
- Tranformation stage : None -> TPS (+1~5%)
- Feature Extrator stage : VGG -> RCNN -> ResNet
- Sequence modeling stage : None -> BiLSTM (+1~2%)
- Prediction stage : CTC -> Attn (+2~5%)

## 1. Transformation Stage

![Figure 11](/assets/images/post/str/figure11.PNG)

## 2. Feature Extration Stage

![Figure 12](/assets/images/post/str/figure12.PNG)

VGG가 가장 시간이 적게 걸리지만 낮은 정확도를 보인다. RCNN은 가장 시간이 걸리지만 가장 적은 메모리와 VGG보다 높은 정확도를 보인다. ResNet이 가장 높은 정확도를 가지지만, 다른 module에 비해 많은 메모리를 필요로 한다. 따라서, 메모리 제약이 존재하면 RCNN이 가장 좋은 trade-off이고, 그렇지 않다면 ResNet을 사용해야 한다. 시간적인 측면에서 세 module 모두 차이가 크지 않으므로 극단적인 경우에만 고려한다.

## 3. Sequence Modeling Stage

![Figure 13](/assets/images/post/str/figure13.PNG)

TPS와 비슷하지만, BiLSTM을 사용하면 비슷한 시간과 메모리에 비해 더 높은 정확도로 향상시킨다.

## 4. Prediction Stage

![Figure 14](/assets/images/post/str/figure14.PNG)

Attn은 CTC에 비해 정확도를 높이려고 할 때 시간이 오래 걸린다.