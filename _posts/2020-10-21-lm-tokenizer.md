---
title: Language Model Tokenizer 이해하기
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
article_tag2: tokenizer
last_modified_at: 2020-10-21 14:47:00 +0800
---

| LM      | type   |text_a | text_b |
|---------|--------|-------|--------|
| None    | -               | A distant person is climbing up a very sheer mountain. | The mountain is unclimbable for humans. |
| BERT    | WordPiece       | '[CLS]', 'a', 'distant', 'person', 'is', 'climbing', 'up', 'a', 'very', 'sheer', 'mountain', '.', '[SEP]' | '[CLS]', 'the', 'mountain', 'is', 'un', '##cl', '##im', '##ba', '##ble', 'for', 'humans', '.', '[SEP]' |
| ALBERT  | SentencePiece   |'[CLS]', '▁a', '▁distant', '▁person', '▁is', '▁climbing', '▁up', '▁a', '▁very', '▁sheer', '▁mountain', '.', '[SEP]' | '[CLS]', '▁the', '▁mountain', '▁is', '▁unc', 'limb', 'able', '▁for', '▁humans', '.', '[SEP]' |
| RoBERTa | (sentencepiece) | 'A', 'Âł', 'd', 'istant', 'Âł', 'person', 'Âł', 'is', 'Âł', 'cl', 'im', 'bing', 'Âł', 'up', 'Âł', 'a', 'Âł', 'very', 'Âł', 'she', 'er', 'Âł', 'mount', 'ain', '.' | 'The', 'Âł', 'mount', 'ain', 'Âł', 'is', 'Âł', 'un', 'cl', 'imb', 'able', 'Âł', 'for', 'Âł', 'humans', '.' |
| T5      | SentencePiece   | '▁A', '▁distant', '▁person', '▁is', '▁climbing', '▁up', '▁', 'a', '▁very', '▁sheer', '▁mountain', '.' | '▁The', '▁mountain', '▁is', '▁un', 'c', 'limb', 'able', '▁for', '▁humans', '.' |

- climbing = climbing
- unblimbable = (이상적) un + climb + able
- unblimbable = (but 실제) un + cl + im + ba + ble / unc + limb + able ...

# Tokenization [Transformer]

## BertTokenizer

Construct a BERT tokenizer based on **WordPiece**.

This tokenizer inherits from **PreTrainedTokenizer** which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

```python
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

## AlbertTokenizer

- Construct an ALBERT tokenizer based on **SentencePiece**.

- This tokenizer inherits from **PreTrainedTokenizer** which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.
```python
from transformers import AlbertTokenizer

albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
```

## RobertaTokenizer

- Constructs a RoBERTa tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

- This tokenizer has been trained to treat spaces like parts of the tokens (a bit like **sentencepiece**) so a word will be encoded differently whether it is at the beginning of the sentence (without space) or not:

```python
from transformers import RobertaTokenizer

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
```

- 띄어쓰기도 하나의 token으로! (id = 232)

## T5Tokenizer

- Construct a T5 tokenizer based on **SentencePiece**.

- This tokenizer inherits from **PreTrainedTokenizer** which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

```python
from transformers import T5Tokenizer

t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
```


