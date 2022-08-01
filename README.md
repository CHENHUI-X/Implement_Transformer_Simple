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
	- 
