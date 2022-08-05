# Implement a simple version of Transformer

## Section 1 : Introduction

- 使用Pytorch简单实现一下Transformer结构,并进行实例测试,具体测试细节见Section 3.
- [主要参考链接](https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch/notebook)
- [Another nice explain and implement for transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- 上述链接提供了Transformer的代码实现,本文主要对细节进行注释.
- README文件中大部分图片都来自源链接,其余图片为网络上收集.

## Section 2 : Transformer

### 2.1 模型整体架构

- ![模型整体架构](https://github.com/CHENHUI-X/Implement_Transformer_Simple/blob/master/img/1_2vyKzFlzIHfSmOU_lnQE4A.png)
- 模型看起来很复杂对不对? 不要慌,我们把他拆开来看.

### 2.2 Self attention

- Transformer的一个很重要机制就是用到了self-attention.它被用在Encoder中,用于解析句子的内容表征.
- Attention机制,不管他具体怎么实现,其本质思想不变,就是**模仿"注意力",找重点内容,关键内容**.
- Why attention ?
    - 想想我们在做图片分类的时候,如果我们的model能够**抓住图片的重点**,那分类的时候是不是会很轻松,毕竟排除了没用到信息.
        
	- ![attention1](https://github.com/CHENHUI-X/Implement_Transformer_Simple/blob/master/img/1__tlq4gNokNM9mhTkz2cEeg.png)
    - 在翻译任务或者机器人QA的时候,如果能快速**抓住句子的重点**并作出应答,也许就不会被人家叫人工智障了吧~
	- ![attention2](https://github.com/CHENHUI-X/Implement_Transformer_Simple/blob/master/img/1_sRy3ukQziKP0TSQqlz3LCg.png)

- How attention ?
	- Attention实现的方式多种多样,花里胡哨.Self-attention只是其中一种.**简单来说,其实就是计算权重,"权"的大小,理解为对不同内容的注意程度**,然后**加权求和得到一个向量**,这个向量一定程度就能够表示**综合主要内容和次要内容的抽象表征.**
	- 细节这里就不给大家详述了,给大家推荐一个很好的Self-attention可视化计算过程. 看完你就明白,Self-attention计算过程其实很简单.
	- [动手Self-attention](https://www.cvmart.net/community/detail/2018)
	- 可以看到,Self-attention吃一个序列,然后吐出相应的vector.因此这样的操作天然适合seq2seq任务,比如翻译任务.

### 2.3 MutiHead-attention
- MutiHEAD是original paper中提到的操作.
- > Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.
	- 有点几个小弟乱拳打死老师傅的意思.
- 具体操作就是,假设embedding dim 为 512 , head 为 8 ,那么每个head操作 512/8 = 64 维的q,k,v. 计算完之后,再把8个head的结果连接合并成一个512维vector,作为最终的attention结果.
	- ![image](https://user-images.githubusercontent.com/55629321/182073415-d89d2626-c100-43f8-a06a-a21452722ffc.png)

### 2.4 Mask-attention
- 这个mask就是字面意思:"挡住,遮挡".这个Mask-attention主要用在decoder中.![image](https://user-images.githubusercontent.com/55629321/182089508-a84a4442-f87e-470d-b357-6ddb0988d1ea.png)

- Why mask ?
	- 在语言翻译任务中,我们输入汉语,输出英语.比如 " 我 爱 你 " - > " I Love You " ,在解析" 我 爱 你 " 的时候我们是需要联系上下文的,也就是说,我们要把输入序列完完整整的做Self-attention,让Encoder充分的抽象出输入的全局表征信息.而在把输出序列放到Decoder中的时候,我们在对 " I Love You " 3个token分别做attention . 要注意,对于 " I " 来说,它应该是翻译任务输出的第1个token,那么这时,他在做attention的时候,只能attention到自己(因为 " Love You " 还没有输出来,应该被忽略[mask] ),而在对 " Love " 做attention的时候,它则可以attention到 " I " 和 " Love " , " You " 则应该被忽略(mask). 

- How mask ?
	- 了解了为什么需要mask,那实现起来其实很简单.我们知道,self-attention 就是计算权重,然后对value加权求和 . 那么假设当前正操作输出序列的第 i 个 token , 只需要把 i+1 之后的 value 对应的权重设置为一个非常小的数(1e-12),然后这样经过softmax之后, i+1 之后的value影响就会变为0,实现了mask 的效果. 具体实现可以见代码.

## Section 3 : Implementation 

### 3.1 Embedding Position 
- 请你回想Self-attention的过程,想想假设第一个输入和最后一个输入换一下位置,最后输出的结果是不是一样的?
	- 答:是一样的,因为对于同一个x, 其相应q,k,v是不变的,和x所处位置无关.
- 由于original paper是将Transformer用到了NLP任务上,比如英语转德语.要知道,语言这种东西,是有前后逻辑的,即**词是具有位置信息的**,而现在的self-attention不能实现.因此,我们要对输入和输出添加位置信息.
- 由于我本身并不是做NLP的,所以对于为什么使用下边的位置编码方式,我不能给出很好的解释.只能给出其公式和一些其他大佬的解读
	- ![image](https://user-images.githubusercontent.com/55629321/182079755-4e63ef75-e738-4b2d-95e0-f42eaacb9da7.png)
	- 几个解读:
		- [如何理解Transformer论文中的positional encoding，和三角函数有什么关系？](https://www.zhihu.com/question/347678607)
		- [Sinusoidal位置编码追根溯源](https://spaces.ac.cn/archives/8231)
- 最后将position embedding 与 input embedding 加起来作为最终输入,进到Encoder.
	- ![image](https://user-images.githubusercontent.com/55629321/182080749-82597447-2741-4561-8da3-4e3e22f712cc.png)

### 3.2 代码部分细节
- 在实现MutiHead-Attention的时候,根据original paper,我们需要对输入X经过3个不同的linner transform得到 q,k,v ,然后再把q,k,v经过linner transformer 之后,去进行attention操作.见下图.
	- ![image](https://user-images.githubusercontent.com/55629321/182082411-47d29c33-452e-4ac3-ba68-a6f583a728b5.png)
- 最后我给出的例子是分析给定句子是Positive or Negative ,那么输入就是句子,输出是1 or 0 ,分别表示Positive or Negative.
- 此外,我们在将句子做word embedding的时候,nn.embedding()函数吃的是句子单词在整个vocabulary中的索引.所以要做一些必要的预处理.具体细节可见代码注释.
- 使用的数据集来源:[Source data ](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
- 由于Transformer 不光要吃输入,还要吃输出,而我给的例子,输出仅仅单纯是0和1,在train model的时候,0 和 1这个输出很容易学到,所以模型准确率为1,指定是有问题的,但是我实在没有精力再去调整了,所以暂时先这样吧,但是Transformer模型架构木大问题.
