---
title: KorNLI and KorSTS
layout: single
author_profile: true
read_time: true
comments: true
share: true
related: true
categories:
- nlp
toc: true
toc_sticky: true
toc_label: 목차
article_tag1: NLP
article_tag2: natural language processing
last_modified_at: 2020-07-14 10:09:00 +0800
---

[KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding](https://arxiv.org/abs/2004.03289) 논문 정리입니다.

dataset github : <https://github.com/kakaobrain/KorNLUDatasets>

## NLI(Natural Language Inference) 자연어 추론

두 문장(premise, hypothesis)을 입력 받아 두 관계를 {entailment, contradiction, neutral}로 classify

### NLI datasets

| dataset | language     | sentence pairs |
|---------|--------------|----------------|
| SNLI    | English      | 570K           |
| MNLI    | English      | 455K           |
| XNLI    | 15 languages |                |

- SNLI (Standard NLI) : based on image captions
- MNLI (Multi-Genre NLI) : from 10 genres
- XNLI (Cross-lingual NLI) : MNLI corpus의 development/test에서 15개의 언어(한국어 포함x)로 확장함

### KorNLI dataset

| KorNLI                     | Total   | Train      | Dev.  | Test  |
| -------------------------- | ------- | ---------- | ----- | ----- |
| Source                     | -       | SNLI, MNLI | XNLI  | XNLI  |
| Translated by              | -       | Machine    | Human | Human |
| \# Examples                | 950,354 | 942,854    | 2,490 | 5,010 |
| Avg. \# words (premise)    | 13.6    | 13.6       | 13.0  | 13.1  |
| Avg. \# words (hypothesis) | 7.1     | 7.2        | 6.8   | 6.8   |



| Example                                                      | English Translation                                          | Label         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------- |
| P: 저는, 그냥 알아내려고 거기 있었어요.<br />H: 이해하려고 노력하고 있었어요. | I was just there just trying to figure it out.<br />I was trying to understand. | Entailment    |
| P: 저는, 그냥 알아내려고 거기 있었어요.<br />H: 나는 처음부터 그것을 잘 이해했다. | I was just there just trying to figure it out.<br />I understood it well from the beginning. | Contradiction |
| P: 저는, 그냥 알아내려고 거기 있었어요.<br />H: 나는 돈이 어디로 갔는지 이해하려고 했어요. | I was just there just trying to figure it out.<br />I was trying to understand where the money went. | Neutral       |

## STS(semantic textual similarity) 텍스트 의미적 유사성

두 문장 사이의 semantic similarity(의미적 유사성)의 정도를 평가함 <br>
0 (dissimilar) <-------> 5 (equivalent)

### STS-B dataset

| dataset | language     | sentence pairs |
|---------|--------------|----------------|
| STS-B   | English      | 8,628          |

- STS-B : image captions, news headlines, user forums

### KorSTS dataset

| KorSTS        | Total | Train   | Dev.  | Test  |
| ------------- | ----- | ------- | ----- | ----- |
| Source        | -     | STS-B   | STS-B | STS-B |
| Translated by | -     | Machine | Human | Human |
| \# Examples   | 8,628 | 5,749   | 1,500 | 1,379 |
| Avg. \# words | 7.7   | 7.5     | 8.7   | 7.6   |



| Example                                                      | English Translation                                      | Label |
| ------------------------------------------------------------ | -------------------------------------------------------- | ----- |
| 한 남자가 음식을 먹고 있다.<br />한 남자가 뭔가를 먹고 있다. | A man is eating food.<br />A man is eating something.    | 4.2   |
| 한 비행기가 착륙하고 있다.<br />애니메이션화된 비행기 하나가 착륙하고 있다. | A plane is landing.<br />A animated airplane is landing. | 2.8   |
| 한 여성이 고기를 요리하고 있다.<br />한 남자가 말하고 있다. | A woman is cooking meat.<br />A man is speaking.      | 0.0   |


## baseline

### Cross-Encoding approaches : NLI, STS

- *de facto* standard approach for NLU task
    - large 모델을 pre-train + 각 task를 fine-tune

- Cross-Encoding approaches
    - pre-train language model은 fine-tuning을 위해 각 setence pair를 하나의 input으로 갖는다.

    1. Korean RoBERTa (base & large)
    pre-train 함

    2. XLM-R (base & large)
    RoBERTa와 동일하지만 vocabulary size가 더 크다(250K) -> embedding 그리고 output layers를 더 크게 만듦

### Bi-Encoding approaches : STS

- NOT pre-trained language models
    1. Korean fastText
        - pre-trained word embedding model
        - sentence embedding = fastText word embeddings의 평균

    2. M-USE (multilingual universal sentence encoder) (base & large)
        - CNN-based sentence encoder model
        - for NLI, question answering, translation ranking
        - 16 languages (Korean 포함)

    - 둘 다 unsupervised STS prediction을 위해 sentence embedding 사이 cosine similarity 계산

- pre-trained language models : SentenceBERT
    - NLI/STS에서 Siames 네트워크 구조로 BERT-like model을 fine-tuning 해야 함

    3. Korean SRoBERTa (base & large)

    4. SXLM-R (base & large)

    - MEAN pooling strategy 채택 : sentence vector = 모든 contextualized word vectors의 평균
