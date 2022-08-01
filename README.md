# Implement a simple version of Transformer

## Section 1 : Introduction

- 使用Pytorch简单实现一下Transformer结构,并进行实例测试,具体测试细节见Section 3.
- [主要参考链接](https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch/notebook)
- 上述链接提供了Transformer的代码实现,本文主要对细节进行注释.
- README文件中大部分图片都来自源链接,其余图片为网络上收集.

## Section 2 : Transformer

### 2.1 模型整体架构

- ![模型整体架构](https://github.com/CHENHUI-X/Implement_Transformer_Simple/blob/master/img/1_2vyKzFlzIHfSmOU_lnQE4A.png)
- 模型看起来很复杂对不对? 不要慌,我们把他拆开来看.

### 2.2 Self attention

- Transformer的一个很重要机制就是用到了self-attention.
- Attention机制,不管他具体怎么实现,其本质思想不变,就是**模仿"注意力",找重点内容,关键内容**.
- Why attention(注意力)?
    - 想想我们在做图片分类的时候,如果我们的model能够**抓住图片的重点**,那分类的时候是不是会很轻松,毕竟排除了没用到信息.
        
	- ![attention1](https://github.com/CHENHUI-X/Implement_Transformer_Simple/blob/master/img/1__tlq4gNokNM9mhTkz2cEeg.png)
    - 在翻译任务或者机器人QA的时候,如果能快速**抓住句子的重点**并作出应答,也许就不会被人家叫人工智障了吧~
	- ![attention2](https://github.com/CHENHUI-X/Implement_Transformer_Simple/blob/master/img/1_sRy3ukQziKP0TSQqlz3LCg.png)

- How attention?
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
)
