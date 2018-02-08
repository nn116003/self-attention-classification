# document classification LSTM + self attention
Pytorch implementation of LSTM classification with self attention. See [A STRUCTURED SELF-ATTENTIVE
SENTENCE EMBEDDING](https://arxiv.org/pdf/1703.03130.pdf)

Some resutls -> [my blog post]()

## IMDB Experiments

training

```
python imdb_attn.py
```

visualize attention
```
python view_attn.py
```
results
./attn.html: <label>\t<pred label>\t sentence with attention(<span ....>)
