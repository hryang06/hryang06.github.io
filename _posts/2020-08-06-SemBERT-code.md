---
title: SemBERT code 돌리기
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
article_tag2: BERT
last_modified_at: 2020-08-06 17:51:00 +0800
---

[Semantics-aware BERT for Language Understanding](https://arxiv.org/abs/1909.02209) 코드 재현한 내용입니다.

github : <https://github.com/cooelf/SemBERT>

run_classifier.py

## commend

### train

```shell
CUDA_VISIBLE_DEVICES=0,1 \
python3 run_classifier.py \
--data_dir glue_labeled_data/SNLI/ \
--task_name snli \
--train_batch_size 16 \
--max_seq_length 128 \
--bert_model bert-large-uncased \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--do_train \
--do_eval \
--do_lower_case \
--max_num_aspect 3 \
--output_dir glue/snli_model_dir
```

### evaluate (using labeled data)

```shell
CUDA_VISIBLE_DEVICES=0,1 \
python3 run_classifier.py \
--data_dir glue_labeled_data/SNLI/ \
--task_name snli \
--eval_batch_size 128 \
--max_seq_length 128 \
--bert_model bert-large-uncased \
--do_eval \
--do_lower_case \
--num_train_epochs 1 \
--max_num_aspect 3 \
--output_dir glue/snli_model_dir
```

### evaluate (using raw data)

```shell
CUDA_VISIBLE_DEVICES=0 \
python3 run_snli_predict.py \
--data_dir glue_data/SNLI \
--task_name snli \
--eval_batch_size 128 \
--max_seq_length 128 \
--max_num_aspect 3 \
--do_eval \
--do_lower_case \
--bert_model snli_model_dir \
--output_dir snli_model_dir \
--tagger_path srl_model_dir
```


## labeled data (input)

example (/SNLI/train.tsv_tag_label)

- idx         : 0
- sentence_a  : A person on a horse jumps over a broken down airplane.
- sentence_b  : A person is training his horse for a competition.
- tag_a
- tag_b
- label       : neutral

```
tag_a
{
    "verbs": 
    [{
        "verb": "jumps", "description": "[ARG0: A person on a horse] [V: jumps] [ARGM-DIR: over a broken down airplane] .",
        "tags": ["B-ARG0", "I-ARG0", "I-ARG0", "I-ARG0", "I-ARG0", "B-V", "B-ARGM-DIR", "I-ARGM-DIR", "I-ARGM-DIR", "I-ARGM-DIR", "I-ARGM-DIR", "O"]
    }], 
    "words": ["A", "person", "on", "a", "horse", "jumps", "over", "a", "broken", "down", "airplane", "."]
}

tag_b
{
    "verbs": 
    [{
        "verb": "is", "description": "A person [V: is] training his horse for a competition .",
        "tags": ["O", "O", "B-V", "O", "O", "O", "O", "O", "O", "O"]
    },
    {
        "verb": "training", "description": "[ARG0: A person] is [V: training] [ARG2: his horse] [ARG1: for a competition] .",
        "tags": ["B-ARG0", "I-ARG0", "O", "B-V", "B-ARG2", "I-ARG2", "B-ARG1", "I-ARG1", "I-ARG1", "O"]
    }],
    "words": ["A", "person", "is", "training", "his", "horse", "for", "a", "competition", "."]
}
```

### class InputExample

example 확인할 때 data.

- guid : id     ex) test-1
- text_a : (string) 1st sentence
- text_b : (string) 2nd sentence, optional(sequence pair task일 때만 필요)
- label : (string) label, optional(test일 때는 필요 없음)

### class InputFeautres

실제 사용하는 data.

- input_ids : BERT 해당 단어 id
    - [CLS] = 101 , [SEP] = 1012 102(문장의 끝을 알리는 듯)
- input_mask : 해당 input을 구분하는 듯, input의 마지막까지 id = 1 (나머지는 0)
- segment_ids : 문장 구분 0/1 ([SEP]까지 부여함)
    - (1 sentence) id = 1
    - (2 sentences) 1st id = 0 , 2nd id = 1

- token_tag_sequence_a <QueryTagSequence>
    - sen_words     : 문장 a의 단어 list (,. 포함 / [CLS], [SEP] 포함X)
    - sen_tags_list : tag a의 tag들의 list (만약 두개가 존재하였다면, 리스트 안에 두개의 리스트로 존재함)
- token_tag_sequence_b
    - sen_words
    - sen_tags_list
- len_seq_a
- len_seq_b
- input_tag_ids
- input_tag_verbs
- input_tag_len
- orig_to_token_split_idx

- label_id
    - NLI : {'contradiction': 0, 'entailment': 1, 'neutral': 2}


## 전처리 (train 전)

### processor 선택 및 설정

dataset(CoLA, MRPC, SST-2, MNLI, QQP, QNLI, RTE, SNLI, WNLI) 마다 존재한다. (STS-B, SQuAD는 없음)

- get_train_examples(data_dir)
- get_dev_examples(data_dir)
- get_labels()

- _create_examples(lines, set_type) : training/dev set example 만드는 method

### tokenizer 설정

wordpiece를 만들어 tokenize가 이뤄진다. 

[convert_examples_to_features()](#convert_examples_to_features())에서 호출하여 사용.

```python
from pytorch_pretrained_bert.tokenization import BertTokenizer

# args.bert_model은 PRETRAINED_VOCAB_ARCHIVE_MAP에서 load되는 .txt (vocab_file)를 말한다.
tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                             do_lower_case=args.do_lower_case)
```

---
#### class BertTokenizer

pre-trained된 BERT를 사용하는 것이 아니라 해당 vocab file을 load하여 tokenize하는데 사용한다.

```python
# pytorch_pretrained_bert/tokenization.py

class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ...
        return ids
    
    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        ...
        return tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file(vocab file) if needed.
        """
        ...
        return tokenizer
```

---

### SRL predictor 설정

Semantic Role Labeler로, [convert_examples_to_features()](#convert_examples_to_features())에서 사용.

```python
from tag_model.tagging import get_tags, SRLPredictor
...
if args.tagger_path != None:
    srl_predictor = SRLPredictor(args.tagger_path)
```

현재 코드에서는 일반적으로 이미 이 작업이 끝난 glue_labeled_data를 사용하기 때문에 사용하지 않지만, raw data를 사용하기 위해서는 tagger_path를 설정하여 SRLPredictor를 사용하도록 한다.

### train_feature 생성

train일 경우 train feature 생성.

```python
train_examples = processor.get_train_examples(args.data_dir)
...
train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer,
            srl_predictor=srl_predictor )
```

---
#### convert_examples_to_features()

[InputExample](#class-inputexample)에서 [InputFeatures](#class-inputfeatures)로 변환하는 함수.

output인 features는 InputFeatures의 list.

```python
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, srl_predictor):
    """Loads a data file into a list of `InputBatch`s."""
    ...
    return features
```

---

### tag 관련 설정

- 현재 모델에서 사용하는 tag 목록 : tag_model.tag_tokenization의 TAG_VOCAB
    - '[PAD]', '[CLS]', '[SEP]'
    - 'B-V', 'I-V'
    - 'B-ARG0', 'I-ARG0', 'B-ARG1', 'I-ARG1', 'B-ARG2', 'I-ARG2', 'B-ARG4', 'I-ARG4'
    - 'B-ARGM-TMP', 'I-ARGM-TMP'
    - 'B-ARGM-LOC', 'I-ARGM-LOC'
    - 'B-ARGM-CAU', 'I-ARGM-CAU'
    - 'B-ARGM-PRP', 'I-ARGM-PRP'
    - 'O'

- tag_tokenizer는 [transform_tag_features()](#transform_tag_features())에서 사용.
    - [BertTokenizer](#class-BertTokenizer)와 비슷하다. BertTokenizer는 해당 단어를 tokenize한다면, TagTokenizer는 tag를 바탕으로 tokenize한다.
- tag_config는 BertForSequenceClassificationTag.from_pretrained()에서 사용.

```python
from tag_model.tag_tokenization import TagTokenizer
from tag_model.modeling import TagConfig

tag_tokenizer = TagTokenizer()
vocab_size = len(tag_tokenizer.ids_to_tags)
print("tokenizer vocab size: ", str(vocab_size)) # 22라고 나옴
tag_config = TagConfig(tag_vocab_size=vocab_size,
                       hidden_size=10,
                       layer_num=1,
                       output_dim=10,
                       dropout_prob=0.1,
                       num_aspect=args.max_num_aspect)
```

#### class TagTokenizer

```python
# tag_model/tag_tokenization.py

TAG_VOCAB = ['[PAD]','[CLS]', '[SEP]', 'B-V', 'I-V', 'B-ARG0', 'I-ARG0', 'B-ARG1', 'I-ARG1', 'B-ARG2', 'I-ARG2', 'B-ARG4', 'I-ARG4', 'B-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-LOC', 'I-ARGM-LOC', 'B-ARGM-CAU', 'I-ARGM-CAU', 'B-ARGM-PRP', 'I-ARGM-PRP', 'O']

class TagTokenizer(object):
    def __init__(self):
        self.tag_vocab = TAG_VOCAB
        self.ids_to_tags = collections.OrderedDict(
            [(ids, tag) for ids, tag in enumerate(TAG_VOCAB)]) # id-tag 연결된 dict

    def convert_tags_to_ids(self, tags):
        """Converts a sequence of tags into ids using the vocab."""
        ...
        return ids

    def convert_ids_to_tags(self, ids):
        """Converts a sequence of ids into tags using the vocab."""
        ...
        return tags
```

### prepare model

실제 사용할 model. 해당 pre-trained BERT model을 불러온다.

```python
from pytorch_pretrained_bert.modeling import BertForSequenceClassificationTag

model = BertForSequenceClassificationTag.from_pretrained(args.bert_model,
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
            num_labels=num_labels,
            tag_config=tag_config)
...
model.to(device)
```

---
#### class BertForSequenceClassificationTag

논문에서 구현한 model로, 실제로 BERT를 포함하여 학습하는 부분이라고 보면 됨.

<https://github.com/huggingface/transformers>에서 제공하는 기본적인 틀은 비슷하지만, pre-trained BERT 이후 layer가 다르다.

```python
# pytorch_pretrained_bert/modeling.py
class BertForSequenceClassificationTag(BertPreTrainedModel):
```

---



### prepare optimiaer

```python
if args.fp16:
   optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          bias_correction=False,
                          max_grad_norm=1.0)
    ...

else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
```


## train

### train feature 설정

```python
train_features = transform_tag_features(args.max_num_aspect, train_features, tag_tokenizer, args.max_seq_length)
```

---
#### transform_tag_features()



```python
def transform_tag_features(max_num_aspect, features, tag_tokenizer, max_seq_length):
    """now convert the tags into ids"""
    ...
    return new_features
```

---

### data load

- torch.tensor()
- train_data = TensorDataset()
- train_sampler = RandomSampler() / DistributedSampler()
- train_dataloader = DataLoader()

### dev data 설정 (train)

한 epoch마다 evaluation 한다. 앞에서 train 설정한거랑 똑같음.

[data to feature](#train_feature-생성)
- eval_examples = processor.get_dev_examples()
- eval_features = convert_examples_to_features()
- eval_features = transform_tag_features()

[data load](#data-load)
- torch.tensor()
- eval_data = TensorDataset()
- eval_sampler = SequentialSampler()
- eval_dataloader = DataLoader()

### N epochs (train)

```python
for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
    model.train()
    ...
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, start_end_idx, 
                                    input_tag_ids, label_ids = batch
        loss = model(input_ids, segment_ids, 
                input_mask, start_end_idx, input_tag_ids,  label_ids)
        
        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        ...
    # Save a trained model
    ...
    output_model_file = os.path.join(
        args.output_dir, str(epoch)+"_pytorch_model.bin")
```

### evaluation (each epoch)

epoch마다 evaluation을 통해 accurach, loss 등 확인할 수 있다.

```python
model_state_dict = torch.load(output_model_file)
predict_model = BertForSequenceClassificationTag.from_pretrained(args.bert_model, 
                                                        state_dict=model_state_dict, 
                                                        num_labels=num_labels, 
                                                        tag_config=tag_config)
predict_model.to(device)
predict_model.eval()
...
output_logits_file = os.path.join(args.output_dir, str(epoch) + "_eval_logits_results.tsv")
with open(output_logits_file, "w") as writer:
    writer.write("index" + "\t" + "\t".join(["logits " + str(i) for i in range(len(label_list))]) + "\n")
        
    for input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        ...
        with torch.no_grad():
            tmp_eval_loss = predict_model(input_ids, segment_ids, 
                    input_mask, start_end_idx, input_tag_ids, label_ids)
            logits = predict_model(input_ids, segment_ids, 
                    input_mask, start_end_idx, input_tag_ids, None)

output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
...
```


## evaluation

뭔가 코드가 중복되는 듯. train에서의 evaluation이랑 거의 동일한 것으로 보임. train을 하는 경우라면 굳이 이 작업을 또 할 필요가 없어 보임.

또 epoch마다 data 설정을 계속 반복하고 있는데 이것도 epoch 밖으로 빼서 반복을 피하는게 좋을 것 같음.

### dev data 설정 (eval)

앞에서 [train에서의 dev data 설정](#dev-data-설정-(train))과 동일하다.

### run prediction for full data

### epoch=1


## prediction

evaluation이랑 prediction이랑 과정이 비슷해서인지 이름이 다소 꼬여있는 듯 하다.

### test data 설정

앞의 [evaluation data 설정](#dev-data-설정-(train))과 동일하다.

[data to feature](#train_feature-생성)
- eval_examples = processor.get_test_examples()
- eval_features = convert_examples_to_features()
- eval_features = transform_tag_features()

[data load](#data-load)
- torch.tensor()
- eval_data = TensorDataset()
- eval_sampler = SequentialSampler()
- eval_dataloader = DataLoader()

### prediction model 과정

test 과정이므로 epoch 없이 한번에 처리한다.

```python
output_model_file = os.path.join(args.output_dir, str(best_epoch)+ "_pytorch_model.bin")
model_state_dict = torch.load(output_model_file)
predict_model = BertForSequenceClassificationTag.from_pretrained(args.bert_model, 
                                                        state_dict=model_state_dict,
                                                        num_labels=num_labels,
                                                        tag_config=tag_config)
predict_model.to(device)
predict_model.eval()

predictions = []
output_logits_file = os.path.join(args.output_dir, str(best_epoch) + "_logits_results.tsv")
with open(output_logits_file, "w") as writer:
    for input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids in tqdm(eval_dataloader, desc="Evaluating"):
        ...
        with torch.no_grad():
            logits = predict_model(input_ids, segment_ids, input_mask,
                                 start_end_idx, input_tag_ids, None)
        logits = logits.detach().cpu().numpy()
        ...

output_test_file = os.path.join(args.output_dir, str(best_epoch) + "_pred_results.tsv")
```

