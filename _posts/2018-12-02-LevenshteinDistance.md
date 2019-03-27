---
layout: post
author: phamdinhkhanh
title: LevenshteinDistance
---

```python
import pandas as pd
data2 = pd.read_excel('SearchItems.xlsx', sheet_name = 'Sheet1', header = 0, encoding = 'ISO-8859-1')
print(data2.shape)
```

    (6100, 2)
    


```python
data2.columns = ['Term', 'Frequence']
data2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Term</th>
      <th>Frequence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pin điện thoại iPhone 6</td>
      <td>123</td>
    </tr>
    <tr>
      <th>1</th>
      <td>điện thoại samsung galaxy a8 plus</td>
      <td>121</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Đồng hồ định vị GPS (kiêm điện thoại trẻ em) I...</td>
      <td>83</td>
    </tr>
    <tr>
      <th>3</th>
      <td>điện thoại cho người già</td>
      <td>78</td>
    </tr>
    <tr>
      <th>4</th>
      <td>điện thoại iphone 6 plus</td>
      <td>67</td>
    </tr>
  </tbody>
</table>
</div>




```python
from underthesea import word_tokenize
dic = []
for item in data2.Term.tolist():
    dic.append(word_tokenize(item))
```

    c:\users\laptoptcc\appdata\local\programs\python\python36\lib\site-packages\sklearn\base.py:251: UserWarning: Trying to unpickle estimator MultiLabelBinarizer from version 0.19.0 when using version 0.20.1. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    c:\users\laptoptcc\appdata\local\programs\python\python36\lib\site-packages\sklearn\base.py:251: UserWarning: Trying to unpickle estimator LabelBinarizer from version 0.19.0 when using version 0.20.1. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    c:\users\laptoptcc\appdata\local\programs\python\python36\lib\site-packages\sklearn\base.py:251: UserWarning: Trying to unpickle estimator SVC from version 0.19.0 when using version 0.20.1. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    c:\users\laptoptcc\appdata\local\programs\python\python36\lib\site-packages\sklearn\base.py:251: UserWarning: Trying to unpickle estimator OneVsRestClassifier from version 0.19.0 when using version 0.20.1. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    


```python
singular_word = list(map(str.split, data2.Term))
dic0 = []
for item in singular_word:
    dic0 += item
# dic1
```


```python
dic1 = []
for i in range(len(dic)):
    dic1 += dic[i]
print('Length of the dictionary: ', len(dic1))
```

    Length of the dictionary:  32923
    


```python
dic2 = dic0 + dic1
```


```python
import numpy as np
dic3 = np.unique(dic2)
print('Length of the dictionary: ', len(dic3))
```

    Length of the dictionary:  4288
    


```python
X = np.unique(dic2, return_counts = True)
data_tf = pd.DataFrame({'Term':X[0], 'Frequence':X[1]})
```


```python
data_tf = data_tf.sort_values('Frequence', ascending = False)
```


```python
data_tf[data_tf.Frequence >= 5].tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Term</th>
      <th>Frequence</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1530</th>
      <td>Thiết</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3020</th>
      <td>ngắt</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3299</th>
      <td>quốc tế</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3927</th>
      <td>vuông</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2884</th>
      <td>miễn</td>
      <td>5</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_tf['length'] = data_tf['Term'].apply(len)
data_tf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Term</th>
      <th>Frequence</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4162</th>
      <td>điện</td>
      <td>7826</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3576</th>
      <td>thoại</td>
      <td>7756</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4164</th>
      <td>điện thoại</td>
      <td>4141</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3387</th>
      <td>samsung</td>
      <td>1172</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4274</th>
      <td>ốp</td>
      <td>810</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_tf[['length', 'Term']].groupby('length').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Term</th>
    </tr>
    <tr>
      <th>length</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>586</td>
    </tr>
    <tr>
      <th>3</th>
      <td>948</td>
    </tr>
    <tr>
      <th>4</th>
      <td>864</td>
    </tr>
    <tr>
      <th>5</th>
      <td>591</td>
    </tr>
    <tr>
      <th>6</th>
      <td>389</td>
    </tr>
    <tr>
      <th>7</th>
      <td>316</td>
    </tr>
    <tr>
      <th>8</th>
      <td>243</td>
    </tr>
    <tr>
      <th>9</th>
      <td>134</td>
    </tr>
    <tr>
      <th>10</th>
      <td>84</td>
    </tr>
    <tr>
      <th>11</th>
      <td>27</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7</td>
    </tr>
    <tr>
      <th>13</th>
      <td>8</td>
    </tr>
    <tr>
      <th>14</th>
      <td>8</td>
    </tr>
    <tr>
      <th>15</th>
      <td>4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_tf = data_tf.loc[(data_tf['length']>=2) & (data_tf['length'] <= 10)]
```


```python
from underthesea import word_tokenize
user_input = 'Điện thoại sansung'
input_split = word_tokenize(user_input)
print(input_split)
```

    ['Điện thoại', 'sansung']
    


```python
import numpy as np
def LevenshteinDistance(s, t, m = None, n = None):
    s = s.lower()
    t = t.lower()
    if m is None:
        m = len(s)
    if n is None:
        n = len(t)
    d = np.zeros(shape = (m+1, n+1))
    d[:, 0] = np.arange(m+1)
    d[0, :] = np.arange(n+1)
    
    for i in np.arange(1, m+1):
        for j in np.arange(1, n+1):
            if s[i-1] == t[j-1]:
                substitution_cost = 0
            else:
                substitution_cost = 1
            d[i, j] = min(d[i-1, j] + 1,                   # deletion
                          d[i, j-1] + 1,                   # insertion
                          d[i-1, j-1] + substitution_cost) # substitution
            
    df = pd.DataFrame(d, columns = [' '] + list(t), index = [' '] + list(s))
#     print(df)
#     ratio_length = m/n
#     if ratio_length < 1: 
#         ratio_length = 1/ratio_length
#     print(d)
#     prior words which have the length nearest with search words
#     yield (max(np.diag(d))+abs(m-n))*ratio_length
#     yield max(np.diag(d))
    return max(np.diag(d)) + abs(m-n)
```


```python
import time
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed
```


```python
@timeit
def LevenshteinDistanceList(string):
    ds = [LevenshteinDistance(string, item) for item in data_tf.Term]
    idxs = np.argsort(ds)
    return data_tf.Term.iloc[idxs]
```


```python
LevenshteinDistanceList('điện thoại')[:2]
```

    'LevenshteinDistanceList'  9634.41 ms
    




    3597    điện thoại
    3557    Điện Thoại
    Name: Term, dtype: object




```python
LevenshteinDistanceList('samsung')[:2]
```

    'LevenshteinDistanceList'  8713.19 ms
    




    1171    SAMSUNG
    2926    samsung
    Name: Term, dtype: object



## Create gram


```python
from nltk import bigrams, trigrams
# list(bigrams(data2['Term'][1]))
```

## II. Fuzzywuzzy search


```python
from fuzzywuzzy import process

@timeit
def _best_guess(input_split):
    best_guess = process.extract(input_split, data_tf.Term)
    return best_guess
_best_guess('Điện thoại')
# print(f"The best match for '{input_split[1]}' is '{best_guess}'")
```

    '_best_guess'  2473.89 ms
    




    [('điện thoại', 100, 3597),
     ('Điện thoại', 100, 3558),
     ('điện Thoại', 100, 3596),
     ('Điện Thoại', 100, 3557),
     ('điện', 90, 3595)]




```python
process.extract('đồng hồ trẻ em', ['đog hồ trẻ em', 'do hồ trẻ em'])
```




    [('đog hồ trẻ em', 89), ('do hồ trẻ em', 82)]




```python
process.extract('đông hồ', data_tf.Term)
```




    [('Đồng hồ', 92, 3572),
     ('đồng hồ', 92, 3658),
     ('ĐỒNG HỒ', 92, 3568),
     ('hồ', 90, 2153),
     ('đèn', 90, 3619)]



## III. Correct spelling


```python
"""
Method 2 : Peter Norvig sur un seul mot
"""

import re
import nltk
from collections import Counter

def words(text): 
    return re.findall(r'\w+', text.lower())

WORDS = Counter(dic2)
```


```python
def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

P('Điện thoại')
```




    0.0001518695137138171




```python
def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = "0123456789abcdefghijklmnopqrstuvwxyz×àáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỹ"
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

len(edits1('Điện thoại'))
```




    2125




```python
def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

len(next(edits2('Điện thoại')))
```




    11




```python
def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

known(['Điện thoại','điện', 'điêns'])
```




    {'Điện thoại', 'điện'}




```python
def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
```


```python
# word = 'điện thaoi'
# known([word])
# edits1(word)
# known(edits2(word))
```


```python
@timeit
def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

correction('điện thoai')
```

    'correction'  0.00 ms
    




    'điện thoại'




```python
from underthesea import word_tokenize
"""
Exemple avec notre text initial 
"""
text1 = "đ"
text2 = "đo"
text3 = "đô"
text4 = "đôn"
text5 = "đồng"
text6 = "đồng hồ"

def common_prefix(prefix, dic = dic1):
    sub_dic = [item for item in dic1 if prefix in item]
    count = Counter(sub_dic)
    return max(count)

def correct_word_in_sentence(text):
#     tokens = word_tokenize(text)
#     r = [correction(token) for token in tokens]
    r = correction(text)
    return r

def suggest_word_search(text):
    r = correct_word_in_sentence(text)
    return common_prefix(r)

tmp = suggest_word_search(text1)
print(tmp)
tmp = suggest_word_search(text2)
print(tmp)
tmp = suggest_word_search(text3)
print(tmp)
tmp = suggest_word_search(text4)
print(tmp)
tmp = suggest_word_search(text5)
print(tmp)
tmp = suggest_word_search(text6)
print(tmp)
# tmp = ' '.join(suggest_word_search(text2))
# print(tmp)
# tmp = ' '.join(suggest_word_search(text3))
# print(tmp)
# tmp = ' '.join(suggest_word_search(text4))
# print(tmp)
# tmp = ' '.join(suggest_word_search(text5))
# print(tmp)
```

    'correction'  0.00 ms
    ưu đãi
    'correction'  0.00 ms
    đỡ
    'correction'  0.00 ms
    đông
    'correction'  0.00 ms
    đencủa
    'correction'  0.00 ms
    đồng hồ
    'correction'  0.00 ms
    đồng hồ
    

## Word2vec model


```python
data2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Term</th>
      <th>Frequence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pin điện thoại iPhone 6</td>
      <td>123</td>
    </tr>
    <tr>
      <th>1</th>
      <td>điện thoại samsung galaxy a8 plus</td>
      <td>121</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Đồng hồ định vị GPS (kiêm điện thoại trẻ em) I...</td>
      <td>83</td>
    </tr>
    <tr>
      <th>3</th>
      <td>điện thoại cho người già</td>
      <td>78</td>
    </tr>
    <tr>
      <th>4</th>
      <td>điện thoại iphone 6 plus</td>
      <td>67</td>
    </tr>
  </tbody>
</table>
</div>




```python
#https://viblo.asia/p/xay-dung-mo-hinh-khong-gian-vector-cho-tieng-viet-GrLZDXr2Zk0
import numpy as np
import matplotlib.pyplot as plt

corpus = ["tôi yêu công_việc .",
          "tôi thích NLP .",
          "tôi ghét ở một_mình"]

words = []
for sentences in corpus:
    words.extend(sentences.split())

words = list(set(words))
words.sort()

X = np.zeros([len(words), len(words)])

for sentences in corpus:
    tokens = sentences.split()
    for i, token in enumerate(tokens):
        if(i == 0):
            X[words.index(token), words.index(tokens[i + 1])] += 1
        elif(i == len(tokens) - 1):
            X[words.index(token), words.index(tokens[i - 1])] += 1
        else:
            X[words.index(token), words.index(tokens[i + 1])] += 1
            X[words.index(token), words.index(tokens[i - 1])] += 1

print(X)
```

    [[0. 1. 1. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 1. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 1.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 1. 0. 1. 0. 1. 0.]
     [0. 0. 1. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 1. 1. 0. 0. 0. 0.]]
    


```python
la = np.linalg
U, s, Vh = la.svd(X, full_matrices=False)

plt.xlim(-1, 1)
plt.ylim(-1, 1)

for i in range(len(words)):
    plt.text(U[i, 0], U[i, 1], words[i])

plt.show()
```


![png]({{"\assets\images\output_38_0.png"}})


# Word2Vec


```python
from gensim.models import Word2Vec
pathdata = 'datatrain.txt'
def read_data(path):
    traindata = []
    sents = open(pathdata, 'r', encoding = 'utf8').readlines()
    for sent in sents:
        traindata.append(sent.split())
    return traindata

train_data = read_data(pathdata)

# train_model

model = Word2Vec(train_data, size=150, window=10, min_count=2, workers=4, sg=1)
model.wv.save("word2vec_skipgram.model")
```

    c:\users\laptoptcc\appdata\local\programs\python\python36\lib\site-packages\gensim\utils.py:1212: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    

# FastText


```python
from gensim.models.fasttext import FastText

train_data = read_data(pathdata)
model_fasttext = FastText(size=150, window=10, min_count=2, workers=4, sg=1)
model_fasttext.build_vocab(train_data)
model_fasttext.train(train_data, total_examples=model_fasttext.corpus_count, epochs=model_fasttext.iter)

model_fasttext.wv.save("fasttext_gensim.model")
```

# Test model


```python
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import KeyedVectors

model = KeyedVectors.load('word2vec_skipgram.model')
```


```python
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(model.wv['diễn_châu'].reshape(1, -1), model.wv['quỳnh_lưu'].reshape(1, -1))
```

    c:\users\laptoptcc\appdata\local\programs\python\python36\lib\site-packages\ipykernel\__main__.py:2: DeprecationWarning: Call to deprecated `wv` (Attribute will be removed in 4.0.0, use self instead).
      from ipykernel import kernelapp as app
    




    array([[0.8237105]], dtype=float32)




```python
model.similarity('diễn_châu', 'quỳnh_lưu')
```

    c:\users\laptoptcc\appdata\local\programs\python\python36\lib\site-packages\gensim\matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int32 == np.dtype(int).type`.
      if np.issubdtype(vec.dtype, np.int):
    




    0.82371056




```python
model.wv.similar_by_word('diễn_châu', topn=10, restrict_vocab=None)
# model.wv['quỳnh_lưu'].shape
```

    c:\users\laptoptcc\appdata\local\programs\python\python36\lib\site-packages\ipykernel\__main__.py:1: DeprecationWarning: Call to deprecated `wv` (Attribute will be removed in 4.0.0, use self instead).
      if __name__ == '__main__':
    c:\users\laptoptcc\appdata\local\programs\python\python36\lib\site-packages\gensim\matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int32 == np.dtype(int).type`.
      if np.issubdtype(vec.dtype, np.int):
    




    [('mông_thành', 0.9094159603118896),
     ('wilfrid', 0.9008690118789673),
     ('bức_bách', 0.8990557789802551),
     ('lý_giác', 0.8972815871238708),
     ('trần_cao', 0.896515965461731),
     ('minh_tiến', 0.8962645530700684),
     ('cùng_đường', 0.8952623009681702),
     ('vũ_ninh', 0.894798755645752),
     ('thọ_châu', 0.8941773772239685),
     ('tôn_thất_thanh', 0.8924148678779602)]




```python
model.wv.similar_by_word('samsung', topn=10, restrict_vocab=None)
```

    c:\users\laptoptcc\appdata\local\programs\python\python36\lib\site-packages\ipykernel\__main__.py:1: DeprecationWarning: Call to deprecated `wv` (Attribute will be removed in 4.0.0, use self instead).
      if __name__ == '__main__':
    c:\users\laptoptcc\appdata\local\programs\python\python36\lib\site-packages\gensim\matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int32 == np.dtype(int).type`.
      if np.issubdtype(vec.dtype, np.int):
    




    [('openoffice', 0.8310470581054688),
     ('microsoft_windows', 0.8298007249832153),
     ('inc', 0.8253514170646667),
     ('data', 0.8169270753860474),
     ('xiaomi', 0.8155237436294556),
     ('macos', 0.8139011263847351),
     ('org', 0.810611367225647),
     ('corp', 0.8075563311576843),
     ('libreoffice', 0.8034858107566833),
     ('galaxy_core', 0.8020549416542053)]




```python
# https://radimrehurek.com/gensim/models/word2vec.html
from gensim.test.utils import common_texts
from gensim.models import Phrases
bigram_transformer = Phrases(common_texts)
model = Word2Vec(bigram_transformer[common_texts], min_count=1)
```

    c:\users\laptoptcc\appdata\local\programs\python\python36\lib\site-packages\gensim\models\phrases.py:598: UserWarning: For a faster implementation, use the gensim.models.phrases.Phraser class
      warnings.warn("For a faster implementation, use the gensim.models.phrases.Phraser class")
    


```python
# https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/
```
