## Pytorch Transformer from scratch

### Architecture
The model is based on the original Transformer architecture from the paper "Attention Is All You Need." 

The key architectural details:
* Number of attention heads: 8
* Number of encoder and decoder blocks: 6
* Feed forward dimension: 2048
* Model hidden dimension: 512
* Dropout rate: 0.1

#### Terms:
* d_model - dimension of the vector
* seq_len - max length of sentence
* d_ff - dimension of the feed forward layer
* h - number of heads
* d_k/d_v - key and value dimension per head

#### Notes: Modified positional encoding for stable numerical computation

---

### Task: Machine Translation

#### Dataset: Opus books

Dataset Source: https://huggingface.co/datasets/Helsinki-NLP/opus_books

To change the language, modify:
* config['lang_src']
* config['lang_tgt']

Note: Check for available language pair on huggingface. Currently using en-it(english to italian).

---

#### Device
To train on mac, set config['device_mps'] = True

Otherwise, cuda or cpu will be used based on availability.

---


### Training and Inference
### Attention Score visualization

---