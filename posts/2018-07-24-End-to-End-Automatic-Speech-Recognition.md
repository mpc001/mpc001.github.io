---
title: End-to-End Automatic Speech Recognition
tags: Deep Learning
---
本文主要介绍了语音识别实现中的一些基本知识点，包括CTC，注意力机制。

# 语音识别

语音识别是一个seq2seq的问题，其表达式如下：
$$
\begin{align*}
    \mathbf{y} = \mathop{\argmax}_{x} logp(\mathbf{y}|\mathbf{x})
\end{align*}
$$
其中输入序列为$\mathbf{x}=(x_1, x_2, ..., x_T)$，输出序列为$\mathbf{y}=(y_1, y_2, ..., y_L)$。在语音识别领域，$\mathbf{x}$指输入的音频序列，$\mathbf{y}$指输出的文字。我们采用encoder-decoder attention框架，在编码部分利用深度网络提取输入信号的high-level特征表达，解码部分包括RNN, Attention和CTC机制。注意力机制作为用作特征的加权，输入raw input信号在网络中完成解码输出，实现端对端的语音识别。

# CTC
CTC全称Connectionist temporal Classification classification，在CTC提出之前，为了训练时序分类任务，我们往往预先将每一帧的输入数据和对应的标签对应，再采用交叉熵作为损失函数进行训练。而CTC免去了每一帧数据和标签的对应信息，只关心整体的输入和与标签是否一致。

解码一串语音序列，最容易想到的方法是把每一帧语音进行识别，然后再合并连续帧出现的重复字符，比如$[c,c,a,a,t]=[c,a,t]$。但是这种方法存在两个缺点：1）无法处理无声的帧信号；2）无法处理单词里有连续重复字母的情况，例如apple。CTC解决了这个问题，它引入了一个记号\textit{blank}，此文用$\epsilon$表示。这一个额外的一个空白字符$epsilon$被强制要求插入到帧和帧之间，在对齐时再移去，举例一些对齐的方式


<center><img src="/images/ctc.svg" width=400></center>

$$
\begin{array}{lllllllll}
\hline 
 &x_1 &x_2 &x_3 &x_4 &x_5 &x_6 &x_7 &output \\
 &a &p &\epsilon &p &l &\epsilon &e &apple\\
 &a &\epsilon &a &p &\epsilon &l &e &aaple\\
\hline
\end{array}
$$

我们可以写出CTC的目标函数：
$$
\begin{align*}
\alpha_t(s) = \sum_{\substack{\pi\in N^T \\ B(\pi_{1:t})=\textbf{l}_{1:s}}}\prod_{t^\prime=1}^ty_{\pi_{t^\prime}}^{t^\prime}
\end{align*}
$$
即$\mathbf{l}_{1:s}$为所有输入长度为$t$的路径的多对一映射，所有路径长度$t$的集合B的概率和为$\alpha_t(s)$。

$$
\begin{align*}
\mathbf{l^\prime}=[\epsilon, y_1, \epsilon, y_2, ..., \epsilon, y_U, \epsilon]
\end{align*}
$$
, 我们计$\alpha_{s,t}$是子序列$\mathbf{z}_{1:s}$在输入长度为$t$时的CTC score。
所以，如果我们知道了$t-1$时刻的CTC score，我们就可以递推出最后时刻的CTC score。

## CTC的概率计算
我们记没有加$\epsilon$的序列$\mathbf{l}$，记在每个有效字符前后加了$\epsilon$的序列为$\mathbf{l^\prime}$。我们将在下面计算得到CTC的概率递推式。

首先考虑初始情况。可以看出，
$$
\begin{align*}
\alpha_1(1) &= y_\epsilon^1 \\
\alpha_1(2) &= y_{\textbf{l}_1}^1 \\
\alpha_1(s) &= 0, \forall s > 2 
\end{align*}
$$
即路径长度为1时，只可能是$\epsilon$，路径长度为2时，只可能是序列$\mathbf{l}$的第一个有效字符。

我们再考虑递推的关系式。CTC是基于条件独立假设，即后一帧的输出只取决于当前帧，与之前帧相互独立。在$t$时刻可能会出现$\epsilon$，也可能刚好出现有效字符。

第一种情况下，
$$
\begin{align*}
\alpha_t(s) = (\alpha_{t-1}(s) + \alpha_{t-1}(s-1))\cdot p_t(\mathbf{l^\prime}_{1:s}|X)
\end{align*}
$$
我们考虑$\mathbf{l^\prime}_{s-1}$在$s-1$时刻无法忽略时，那么1) $\mathbf{l^\prime}_{s-1}$可能为一个有效字符，鉴于每两个有效字符之间存在一个额外的空白字符，所以可以推出$\mathbf{l^\prime}_{s}=\epsilon$; 2) $\mathbf{l^\prime}_{s-1}$可能为空白字符，则$\mathbf{l^\prime}_{s}$和$\mathbf{l^\prime}_{s-2}$为重复的字符。

第二种情况下下，前一个字符可以跳过，则这种情况下$\mathbf{l^\prime}_{s-1}$为一个空白字符$\epsilon$，而$\mathbf{l^\prime}_{s-2}$和$\mathbf{l^\prime}_{s}$均为有效字符。
$$
\begin{align*}
\alpha_t(s) = (\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \alpha_{t-1}(s-2)) \cdot p_t(\mathbf{l^\prime}_{1:s}|X)
\end{align*}
$$
通过递推表达式，我们可以表示出整串序列的表达式，
$$
\begin{align*}
p({\mathbf{l}|\mathbf{x}}) = \alpha_T(|\mathbf{l^\prime}|) + \alpha_T(|\mathbf{l^\prime}|-1)
\end{align*}
$$

# 训练方式
本文在注意力解码模型(Attention Decoder)的基础上结合CTC，目的是为了利用CTC用来帮助对齐长序列。因此，目标函数可以写作下式：
$$
\begin{align*}
\mathcal{L} = \alpha \mathcal{L}^{ctc} + (1 - \alpha)\mathcal{L}^{att}
\end{align*}
$$
其中，可调节的参数$\alpha\in[0, 1]$作为两个任务的相对权重。

# 注意力
注意力（Attention）是一种权重的矩阵，根据Attention is all you need的定义，An attention function can be described as mapping a query and a set of key-value pairs to an output。其中，输出由加权的value得到，权重根据query和相对应的key通过函数计算得到。

基于内容的注意力机制的表达式的变量中包括$h$和$s_{i-1}$。其表达式为
$$
\begin{align*}
\alpha_i = Attend(s_{i-1}
, h_j)
\end{align*}
$$

基于位置的注意力机制表达式的变量包括$s_{i-1}$和$\alpha_{i-1}$。其表达式为
$$
\begin{align*}
\alpha_i=Attend(s_{i-1}, \alpha_{i-1})
\end{align*}
$$

Attention机制通过概率的链式法则直接估计后验概率
$$
\begin{align*}
p(\mathbf{y}|\mathbf{x})=\prod_{l=1}^Lp(y_l|y_1,...,y_{l-1}, \mathbf{x})
\end{align*}
$$
其中$p(y_l|y_1,...,y_{l-1}, \x)$通过下式得到
$$
\begin{align*}
    h_t &= Encoder(X) \\
    a_{lt}&= LocationAttention(\{a_{l-1}\}_{t=1}^T, q_{l-1}, h_t) \\
    r_l &= \sum_{t=1}^Ta_{lt}h_t
\end{align*}
$$
解码结合了注意力机制和CTC机制，相比较单一的注意力机制模型，该解码不仅考虑到对齐的单调性，也考虑到了解码过程中容易过早检测到end of sentence 符号结束而失去准确性。

# 语言模型
在推断上我们加入递归神经网络语言模型（RNN-LM），相比较$n$-gram 语言模型（$n$-gram LM），递归神经网络语言模型可以更好地降低错词率。
语言模型的概率和CTT/Attention的概率结合作为推断下一个字符的分数。其表达式可以写作：

# 参考文献
<sub>A. Hannun, “Sequence modeling with ctc.”</sub>

<sub>Chorowski, Jan K., et al. "Attention-based models for speech recognition."</sub>

<sub>Watanabe, Shinji, et al. "Hybrid CTC/attention architecture for end-to-end speech recognition."</sub>

<sub>Graves, Alex, et al. "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks."</sub>
