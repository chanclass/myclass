# Keras的Tokenizer源码

非常喜欢`keras`框架，平时都是使用封装好的API，基本完全可以满足需求，很少需要修改源码的。最近对keras的实现更加好奇了，于是花点时间读源码，然后整理点学习笔记吧。

我大致浏览了keras中文文档以及英文文档和源码，发现文档不太全面，很多源码实现的接口而文档中没有涉及到，于是萌生了自己整理分析源码的想法。

本文作为第一篇文档，先从预处理的`tokenizer`开始整理。

## tokenizer是什么

计算机在处理语言文字时，是无法理解文字的含义，通常会把一个词（中文单个字或者词组认为是一个词）转化为一个正整数，于是一个文本就变成了一个序列。而`tokenizer`的核心任务就是做这个事情。

## 基本参数说明

```python
keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
```

- **num_words**: the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.
- **filters**: a string where each element is a character that will be filtered from the texts. The default is all punctuation, plus tabs and line breaks, minus the ' character.
- **lower**: boolean. Whether to convert the texts to lowercase.
- **split**: str. Separator for word splitting.
- **char_level**: if True, every character will be treated as a token.
- **oov_token**: if given, it will be added to word_index and used to replace out-of-vocabulary words during text_to_sequence calls[^2]

--------------------------------------------------------------------------------

- **num_words**: 保留的最大词数，根据词频计算，保留前`num_word - 1`个
- **filters**: 过滤器，默认过滤掉常用的特殊符号
- **lower**：是否转化为小写
- **split**：词的分隔符
- **char_level**：是否将每个字符都认为是词，默认是否。在处理中文时如果每个字都作为是词，这个参数改为`True`.
- **oov_token**：如果给出，会添加到词索引中，用来替换超出词表的字符
- **document_count**：文档个数，这个参数一般会根据喂入文本自动计算，无需给出

## 几个重要接口

这里我直接截图了keras的中文文档[^1]。有一个小问题，这是对象或者实例的方法，而不是类方法。

![](http://www.blackedu.vip/admin/wp-content/uploads/2019/09/屏幕快照-2019-09-15-08.19.07.png)

![](http://www.blackedu.vip/admin/wp-content/uploads/2019/09/屏幕快照-2019-09-15-08.19.16.png)

## 源码分析

```python
def fit_on_texts(self, texts):
        """Updates internal vocabulary based on a list of texts.
        基于文本列表，更新内部词典，主要是word_index,和index_word这两个属性

        In the case where texts contains lists,
        we assume each entry of the lists to be a token.

        Required before using `texts_to_sequences` or `texts_to_matrix`.

        # Arguments
            texts: can be a list of strings,
            字符串列表
                a generator of strings (for memory-efficiency),
                字符串的生成器
                or a list of list of strings.
                列表中嵌套的列表字符串
        """

        for text in texts:
            self.document_count += 1 # 更新文档数
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text] # 将所有字符转为小写
                    else:
                        text = text.lower()
                seq = text # seq存储文本的词序列，单个字或者词作为元素
            else:
                seq = text_to_word_sequence(text,
                                            self.filters,
                                            self.lower,
                                            self.split) # 文本转为词序列，这个接口单独分析
            # self.word_counts是一个有序字典，用来统计词频
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                # In how many documents each word occurs
                self.word_docs[w] += 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True) # 按照词频降序排序
        # forcing the oov_token to index 1 if it exists
        # 强制把oov_token的索引设置为1，0通常是padding的补充值
        # 是否指定超出词典的标记
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # note that index 0 is reserved, never assigned to an existing word
        # 更新word_index
        self.word_index = dict(
            list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
        # 更新index_word
        self.index_word = dict((c, w) for w, c in self.word_index.items())

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c
```

接口的实现思路总结：将输入的文本列表先拆成词，然后统计每个词的词频，并存入有序字典中。将字段元素转为列表，并且降序排列。根据这个排序的列表可以得到word_index和index_word。之后的把文本转为词序列`texts_to_sequences`或者把词序列转为文本`sequences_to_texts`，依赖这两个词表。

```python
    def texts_to_sequences_generator(self, texts):
        """Transforms each text in `texts` to a sequence of integers.

        Each item in texts can also be a list,
        in which case we assume each item of that list to be a token.

        Only top `num_words-1` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of texts (strings).

        # Yields
            Yields individual sequences.
        """
        num_words = self.num_words # 保留最常用的词数
        oov_token_index = self.word_index.get(self.oov_token) # 获取oov_token的词索引
        for text in texts:
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = text_to_word_sequence(text,
                                            self.filters,
                                            self.lower,
                                            self.split)
            vect = [] # 存储返回结果
            for w in seq:
                # 注意这里的word_index是根据词频的降序排列的
                i = self.word_index.get(w) # 获取词索引
                if i is not None: # 拿到了词索引
                    # 指定了num_words 并且词索引大于num_words
                    if num_words and i >= num_words:
                        if oov_token_index is not None: # oov_token 的词索引不为空
                            vect.append(oov_token_index) # 将这个词当成 oov_token
                    else:
                        vect.append(i) # 没有指定num_words或者i<num_words 加入
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            yield vect # 生成器的返回
            # 这里有个问题，没有指定num_words或者i<num_words ，此时也没有指定oov_token，那么这个词将会被忽略
```

接口的实现思路总结：获取到词索引，然后判断是否满足返回条件。

- 如果词索引没有拿到，会试图用oov_token填充；如果oov_token也没有指定，那就直接忽略掉
- 拿到词索引，判读是否指定num_words，以及词索引是否大于num_words

`texts_to_sequences`底层直接调用了这个生成器。

```python
    def sequences_to_texts_generator(self, sequences):
        """Transforms each sequence in `sequences` to a list of texts(strings).

        Each sequence has to a list of integers.
        In other words, sequences should be a list of sequences

        Only top `num_words-1` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            sequences: A list of sequences.

        # Yields
            Yields individual texts.
        """
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for seq in sequences:
            vect = []
            for num in seq:
                word = self.index_word.get(num) # 根据词索引获取到词
                if word is not None: # 如果词不为空
                    if num_words and num >= num_words: # num_words指定了并且词索引大于等于num_words
                        if oov_token_index is not None: # 指定了oov_token
                            vect.append(self.index_word[oov_token_index]) # 这个词就是 oov_token
                    else:
                        vect.append(word) # 没指定oov_token 或者num < num_words
                elif self.oov_token is not None: # word 为空 但是oov_token 不为空
                    vect.append(self.index_word[oov_token_index])# 这个词也是 oov_token
            vect = ' '.join(vect) # 词序列拼接成字符串
            yield vect
```

接口分析：实现思路在注释基本清楚了。`sequences_to_texts`直接调用了这个生成器。

```python
    def get_config(self):
        '''Returns the tokenizer configuration as Python dictionary.
        The word count dictionaries used by the tokenizer get serialized
        into plain JSON, so that the configuration can be read by other
        projects.

        # Returns
            A Python dictionary with the tokenizer configuration.
        '''
        json_word_counts = json.dumps(self.word_counts)
        json_word_docs = json.dumps(self.word_docs)
        json_index_docs = json.dumps(self.index_docs)
        json_word_index = json.dumps(self.word_index)
        json_index_word = json.dumps(self.index_word)

        return {
            'num_words': self.num_words,
            'filters': self.filters,
            'lower': self.lower,
            'split': self.split,
            'char_level': self.char_level,
            'oov_token': self.oov_token,
            'document_count': self.document_count,
            'word_counts': json_word_counts,
            'word_docs': json_word_docs,
            'index_docs': json_index_docs,
            'index_word': json_index_word,
            'word_index': json_word_index
        }
```

```python
   def to_json(self, **kwargs):
        """Returns a JSON string containing the tokenizer configuration.
        To load a tokenizer from a JSON string, use
        `keras.preprocessing.text.tokenizer_from_json(json_string)`.

        # Arguments
            **kwargs: Additional keyword arguments
                to be passed to `json.dumps()`.

        # Returns
            A JSON string containing the tokenizer configuration.
        """
        config = self.get_config()
        tokenizer_config = {
            'class_name': self.__class__.__name__,
            'config': config
        }
        return json.dumps(tokenizer_config, **kwargs)
```

接口分析：`to_json`是把`tokenizer`对象序列化，并且以json的格式存储起来。存储以后肯定要提供一个接口来反序列化得到`tokenizer`，这个反序列的接口是`tokenizer_from_json`.

```python
def tokenizer_from_json(json_string):
    """Parses a JSON tokenizer configuration file and returns a
    tokenizer instance.

    # Arguments
        json_string: JSON string encoding a tokenizer configuration.

    # Returns
        A Keras Tokenizer instance
    """
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get('config')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))

    tokenizer = Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word

    return tokenizer
```

## 总结

本文大致分析了keras的Tokenizer类中比较重要的参数，属性以及对象的方法。这个分词器主要是把文本转化为词序列，同时也提供了词序列转为文本的接口。源码非常清晰简洁，功能基本完善，如果需要实现部分定制化的功能，继承这个类，添加一些接口也非常简单。比如我需要删除低频词而不是设置保留词。在面对大量文本时，保留词的个数很难确定，具体是2万还是1.5万不好设置，但是对于低频词是容易界定的。

[^1]: https://keras-cn-docs.readthedocs.io/zh_CN/latest/preprocessing/text/
[^2]: https://keras.io/preprocessing/text/
