## Seq2Seq Demo

This repository includes some demo Seq2Seq models.

Note: The project refers to <https://github.com/bentrevett/pytorch-seq2seq>

<br/>

Datasets:

* `dataset1`: [news-commentary-v14.de-en](http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.de-en.tsv.gz)

Models:

* `model1`: [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
* `model2`: [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
* `model3`: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* `model4`: 
* `model5`: [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)
* `model6`: [Attention is All You Need](https://arxiv.org/abs/1706.03762)

### Process Data

```shell
PYTHONPATH=. python dataprocess/process.py
```

### Unit Test

* For loader

```shell
# loader1
PYTHONPATH=. python loaders/loader1.py
# loader2
PYTHONPATH=. python loaders/loader2.py
```

* For module

```shell
# module1
PYTHONPATH=. python modules/module1.py
# module2
PYTHONPATH=. python modules/module2.py
# module3
PYTHONPATH=. python modules/module3.py
# module4
PYTHONPATH=. python modules/module4.py
# module5
PYTHONPATH=. python modules/module5.py
# module6
PYTHONPATH=. python modules/module6.py
```

### Main Process

```shell
python main.py
```

You can change the config either in the command line or in the file `utils/parser.py`

Here are the examples for each module:

```shell
# module1
python main.py \
    --module 1 \
    --grad_clip 1 \
    --rnn_type lstm \
    --enc_emb_dim 256 \
    --dec_emb_dim 256 \
    --enc_hid_dim 512 \
    --dec_hid_dim 512 \
    --enc_n_layers 2 \
    --dec_n_layers 2 \
    --enc_n_directions 1 \
    --dec_n_directions 1 \
    --enc_dropout 0.5 \
    --dec_dropout 0.5
```

```shell
# module2
python main.py \
    --module 2 \
    --grad_clip 1 \
    --rnn_type gru \
    --enc_emb_dim 256 \
    --dec_emb_dim 256 \
    --enc_hid_dim 512 \
    --dec_hid_dim 512 \
    --enc_n_layers 1 \
    --dec_n_layers 1 \
    --enc_n_directions 1 \
    --dec_n_directions 1 \
    --enc_dropout 0.5 \
    --dec_dropout 0.5
```

```shell
# module3
python main.py \
    --module 3 \
    --grad_clip 1 \
    --rnn_type gru \
    --enc_emb_dim 256 \
    --dec_emb_dim 256 \
    --enc_hid_dim 512 \
    --dec_hid_dim 512 \
    --enc_n_layers 1 \
    --dec_n_layers 1 \
    --enc_n_directions 2 \
    --dec_n_directions 1 \
    --enc_dropout 0.5 \
    --dec_dropout 0.5
```

```shell
# module4
python main.py \
    --module 4 \
    --grad_clip 1 \
    --rnn_type gru \
    --enc_emb_dim 256 \
    --dec_emb_dim 256 \
    --enc_hid_dim 512 \
    --dec_hid_dim 512 \
    --enc_n_layers 1 \
    --dec_n_layers 1 \
    --enc_n_directions 2 \
    --dec_n_directions 1 \
    --enc_dropout 0.5 \
    --dec_dropout 0.5
```

```shell
# module5
python main.py \
    --module 5 \
    --grad_clip 0.1 \
    --enc_emb_dim 256 \
    --dec_emb_dim 256 \
    --enc_hid_dim 512 \
    --dec_hid_dim 512 \
    --enc_filter_layers 10 \
    --dec_filter_layers 10 \
    --enc_kernel_size 3 \
    --dec_kernel_size 3 \
    --enc_dropout 0.25 \
    --dec_dropout 0.25
```

```shell
# module6
python main.py \
    --module 6 \
    --grad_clip 1 \
    --enc_hid_dim 256 \
    --dec_hid_dim 256 \
    --enc_transformer_layers 3 \
    --dec_transformer_layers 3 \
    --enc_attention_heads 8 \
    --dec_attention_heads 8 \
    --enc_mid_dim 512 \
    --dec_mid_dim 512 \
    --enc_dropout 0.1 \
    --dec_dropout 0.1
```

