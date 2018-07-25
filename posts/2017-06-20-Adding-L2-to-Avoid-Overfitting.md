---
title: Adding L2 regularization to Avoid Overfitting
tags: Deep Learning
---

因为保留的特征变量太多，远大于数据量，这样得到候选的模型太多，不太可能产生拟合很好的模型，所以会产生过拟合的现象，过拟合意味着低偏差，高方差，即模型预期的输出结果和真实结果的差距（训练集）减小时，但是训练出的模型在测试集的误差增大。为了避免过拟合，可以采取：1）增加数据量，2）引入正则项，3）引入Dropout层。这里解释为什么引入L2正则项可以避免过拟合。

# 先验解释
Maximum Likelihood Estimation (MLE)相比较 Maximum A-Posterior Estimation (MAP)的区别就是引入了先验，可以当作正则项。

- MLE: 
$$
\begin{align*}
\theta_{MLE} = (\mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{y}
\end{align*}
$$
- MAP: 
$$
\begin{align*}
\theta_{MAP} &= (\mathbf{\Phi}^T\mathbf{\Phi} + \frac{\sigma^2}{b^2}\mathbf{I})^{-1}\mathbf{\Phi}^T\mathbf{y} \\
\theta_{MAP} &= (\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I})^{-1}\mathbf{\Phi}^T\mathbf{y}
\end{align*}
$$
这里取$$\lambda=\frac{\sigma^2}{b^2}$$
如果特征数大于样本数，$\mathbf{\Phi}^T\mathbf{\Phi}$不是满秩，无法直接求逆，也就是就是数据少，不足以确定一个解，换句话说，就是存在很多解，在很多解中选到最好的解的可能性小，就会过拟合。如果引入$\lambda\mathbf{I}$,满秩逆矩阵存在唯一解。引入正则后我们分析权重$\theta$的变化:引入正值$\lambda$可以使权重减小，进而模型的复杂度降低，即可选模型数量减少，更容易产生拟合很好的模型，也就不容易过拟合。


# 定量解释
下式为引入正则项的代价函数，第一项$C_0$为原代价函数，第二项为正则项。
$$
\begin{align*}
C = C_0 + \frac{\lambda}{2n}\sum_ww^2
\end{align*}
$$
可以看出，为了最小化代价函数，如果$\lambda$较小，偏向于最小化代价函数，如果$\lambda$较大，偏向于小的权值。原代价函数在这里可以看作引入$\lambda=0$的正则项，此时$w$的协方差($\mathbf{w}^T\mathbf{w}$)就可以无穷大。
$$
\begin{align*}
\frac{\partial C}{\partial w} &= \frac{\partial C_0}{\partial w} + \frac{\lambda}{n}w \\
\frac{\partial C}{\partial b} &= \frac{\partial C_0}{\partial b}
\end{align*}
$$
在这里，我们采用随机梯度下降，假设$\eta$为步长，则对于常数项$b$不会发生变化，对于权值项$w$则衰减更快。更小的参数值意味着模型的复杂度越低。
$$
\begin{align*}
b &= b - \eta\frac{\partial C_0}{\partial b}\\
w &= w - \eta(\frac{\partial C_0}{\partial w} + \frac{\lambda}{n}w)\\
w &= (1-\eta\frac{\lambda}{n})w - \eta\frac{\partial C_0}{\partial w}
\end{align*}
$$


# 为什么大的权值模型复杂，小的权值模型简单

小的权值意味着更通用化的表达, 网络的行为不会因为我们随意更改了一些输入而改变太多，不容易学习到局部的噪声。与之相对的是，如果输入有一些小的变化，一个拥有大权重的网络会大幅改变其行为来响应变化。因此一个未正则化的网络可以利用大权重来学习得到训练集中包含了大量噪声信息的复杂模型。