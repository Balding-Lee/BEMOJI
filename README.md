# BEMOJI

## Requires

python = 3.7.0

pytorch = 1.10.0

transformers = 4.17.0

numpy = 1.21.5

scikit-learn = 0.23.2

scipy = 1.7.3

## How to use

- if you want to pre-train BEMOJI:

```python
cd run_models
python pre_train_BEMOJI.py -m pre_train_chinese
python pre_train_BEMOJI.py -m pre_train_english
```

`pre_train_chinese` stands for pre-train BEMOJI on Chinese corpus, `pre_train_english` stands for pre-train BEMOJI on Github corpus.

- if you want to pre-train other transformer-based modes:

```python
cd run_models
python pre_train_transformers.py --model BERT_base --mode pre_train_chinese
```

- if you want to pre-train DeepMoji:

```python
cd run_models
python pre_train_DeepMoji.py -m pre_train_chinese
```

- if you want to fine-tune BEMOJI on downstream tasks (with emojis):

```python
cd run_models
python BEMOJI_for_SA.py -d chinese -p 10 -f 1 -a None
```

- if you want to fine-tune BEMOJI on downstream tasks (without emojis):

```python
cd run_models
python BEMOJI_for_nonemoji_SA.py -d SemEval -p 10
```

- if you want to fine-tune transformers on downstream tasks (with emojis):

```python
cd run_models
python run_bert.py -m BERT_base -d chinese -p 10 -f 1
```

- if you want to fine-tune transformers on downstream tasks (without emojis):

```python
cd run_models
python run_bert_nonemoji_SA.py -d SemEval
```

- if you want to fine-tune DeepMoji on downstream tasks:

```python
cd run_models
python DeepMoji_for_SA.py -m Weibo
```

- if you want to study some cases:

```python
cd run_models
python case_study.py -d Chinese --k 5 -m BEMOJI -t '你站在桥上看风景，看风景的人在楼上看你'
```

## Additional information

Some files are very large, so I put them on Baidu network disk: https://pan.baidu.com/s/1lQglRSPjZu7AFa2QYGIgAQ, with the extraction code: MOJI .

The files in `model_parameters` belong to `models > model_parameters`.

The files in `pretrain_parameters` belong to `models > pretrain_parameters`.

The file in `pre-train_data` belongs to `root > pre-train_data`.

The files in `word2vec` belong to `static_data > word2vec`.
