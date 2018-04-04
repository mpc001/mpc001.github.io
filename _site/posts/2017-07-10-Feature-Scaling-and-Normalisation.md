---
title: Feature Scaling and Normalisation
tags: Deep Learning
---
假设一种情况，即数据的训练需要用到多个特征，特征和特征间的数值范围相差大，那么在训练过程中数值范围大的特征的梯度会因为存在数值范围小的特征的约束而变得很小，对于整个模型的训练来说，训练过程就会变慢。特征间数值范围的差异会对训练的速度和精度影响。本文介绍两种特征缩放的常用方法，分别为标准化（Z-score normlisation）和Min-Max缩放（Min-Max scaling）。

# Z-score normalisation
Z-score normalisation即将每个特征的值进行标准化。在SVM和神经网络里常用该方法。
$$
X'=\frac{X-X^{train}_\mu}{X^{train}_\sigma} 
	$$

# Min-Max scaling
Min-Max缩放即将一个特征向量将会被缩放到指定的$(X_{max}, X_{min})$，在scikit-learn里，默认会将缩放的范围限定在(0, 1)。
$$X' = \frac{X-X^{train}_{min}}{X^{train}_{max}-X^{train}_{min}}\times(X_{max}-X_{min})+X_{min} $$

#为什么要特征缩放
特征缩放即让特征进行标准化，其目的是为了让所有的特征具有同等的重要性。一些算法需要特征缩放，例如K-均值聚类和SVM，SVM计算点到线的距离，K-means计算点到点的距离。当其中一个特征比其他特征拥有更大的数值范围时，算法的精度会受到该特征的影响。


