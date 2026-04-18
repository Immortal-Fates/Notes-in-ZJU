# HuggingFace Intro

## Intro

check the References[1]

一个完整的transformer模型主要包含三部分：

1. **Config**，控制模型的名称、最终输出的样式、隐藏层宽度和深度、激活函数的类别等。将Config类导出时文件格式为 json格式，就像下面这样：

   ```json
   {
     "attention_probs_dropout_prob": 0.1,
     "hidden_act": "gelu",
     "hidden_dropout_prob": 0.1,
     "hidden_size": 768,
     "initializer_range": 0.02,
     "intermediate_size": 3072,
     "max_position_embeddings": 512,
     "num_attention_heads": 12,
     "num_hidden_layers": 12,
     "type_vocab_size": 2,
     "vocab_size": 30522
   }
   ```

   当然，也可以通过config.json来实例化Config类，这是一个互逆的过程。

2. **Tokenizer**，这是一个将纯文本转换为编码的过程。注意，Tokenizer并不涉及将词转化为词向量的过程，仅仅是将纯文本分词，添加[MASK]标记、[SEP]、[CLS]标记，并转换为字典索引。Tokenizer类导出时将分为三个文件，也就是：

   - vocab.txt

     词典文件，每一行为一个词或词的一部分

   - special_tokens_map.json 特殊标记的定义方式

     ```json
     {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", 
      "cls_token": "[CLS]", "mask_token": "[MASK]"}
     ```

   - tokenizer_config.json 配置文件，主要存储特殊的配置。

3. **Model**，也就是各种各样的模型。除了初始的Bert、GPT等基本模型，针对下游任务，还定义了诸如`BertForQuestionAnswering`等下游任务模型。模型导出时将生成`config.json`和`pytorch_model.bin`参数文件。前者就是1中的配置文件，这和我们的直觉相同，即config和model应该是紧密联系在一起的两个类。后者其实和torch.save()存储得到的文件是相同的，这是因为Model都直接或者间接继承了Pytorch的Module类。从这里可以看出，HuggingFace在实现时很好地尊重了Pytorch的原生API。

## Practice

check the References[2],里面有如何进行huggingface模型加载和下游人物使用



## References

1. [知乎-huggingface超详细介绍](https://zhuanlan.zhihu.com/p/535100411)
2. [HuggingFace-transformers系列的介绍以及在下游任务中的使用](https://www.cnblogs.com/dongxiong/p/12763923.html)
