---
title: Fundamentals in Speech Enhancement
tags: Deep Learning
---
本文列举了在做语音增强的一些基本知识点，包括特征提取、噪音实现和评价指标。

# 短时傅里叶变换
短时傅里叶变换的定义如下：
$$
X(N, \omega) = \sum_{m=-\infty}^\infty x(m)w(n-m)e^{-j\omega m}
$$
其中，$x(m)$是输入信号，$w(m)$是分析窗，其中包含了$n$个采样点。短时傅立叶变换实质是对离散时间$n$和连续频率$w$进行的变换操作。

我们同样可以类比得到离散的短时傅里叶变换形式，即频率$w$在频率空间中等间隔采样N个，离散的短时傅里叶变换定义如下：
$$
X(N, \omega_k) = X(n, k) = \sum_{m=-\infty}^\infty x(m)w(n-m)e^{-j\frac{2\pi}{N}km}
$$
如果固定频率值，离散的短时傅里叶变换可转为
$$
X(N, \omega_k) = \sum_{m=-\infty}^\infty [x(m)e^{-j\omega_km}]w(n-m) = [x(n)e^{-j\omega_kn}]\circledast w(n)
$$    
其中，$\circledast$为卷积操作。该式认为短时傅里叶变换的过程为输入信号$x(n)$经过频域的变换后再经过分析窗$w(n)$得到。

*spectrogram is a graphical display of the power spectrum of speech as a function of time*, 定义式为$S(n, \omega) = |X(n, \omega)|^2$。

# 信噪比
信噪比为干净信号能量与加入噪声能量的比值，信噪比的定义如下：
$$
SNR(dB) = 10log_{10}(\frac{P_{signal}}{P_{noise}})
$$
信噪比的python实现形式如下：

```python
def getPower(clip):
    clip2 = clip.copy()
    clip2 = clip2 **2
    return np.sum(clip2) / (len(clip2) * 1.0)

def addNoise(data, noise, snrTarget):
    sigPower = getPower(data)
    noisePower = getPower(noise)
    factor = (sigPower / noisePower ) / (10**(snrTarget / 10.0))
    data = (data + noise*np.sqrt(factor)).astype(np.float32)
    return data
```

# 评价指标
评价降噪的能力可以由主观评价指标和客观评价指标进行评价。

主观评价语音质量的标准用来衡量听话时对语音质量的主观感觉。在该评价标准中，语音质量被分为5个等级：优，良，正常，差，劣。由所有倾听者评级并取平均值，该值被称为平均评定得分Mean Opinion Score(MOS)。通常MOS分为4.0-4.5为高质量语音编码，MOS3.5被称为通信质量，MOS低于3分为合成语音质量。

客观评价标准例如分段信噪比和PESQ计算。
由于用信噪比衡量主观语音质量很差，一种客观的评价标准为分段信噪比，此方法基于帧来实现。
$$
\begin{align*}
d_{SEGSNR}=\frac{10}{M}\sum^{M-1}_{m=0}log\frac{\sum_{n=Nm}^{Nm+N-1}s_\phi^2(n)}{\sum_{n=Nm}^{Nm+N-1}[s_d(n)-s_\phi(n)]^2}
\end{align*}
$$
其中第$n$个片段中，$s_\phi(n)$是清晰信号，$s_d(n)$是嘈杂信号，$N$是帧长，$M$是帧数。

另一种评价指标，PESQ全称Perceptual evaluation of speech quality，即主观语音质量评估，该值与平均评定得分存在很强的相关性，因此常被用作评定语音质量。PESQ通过时域上的对称损伤密度和非对称损伤密度来衡量，在采样率16KHz下的信号，帧损伤被定义为20帧(320ms)的损伤密度的综合值，先取L6范数，再取L2范数。
取L6范数
$$
\begin{align*}
D_k^{''} &=(\frac{1}{20}\sum_{n=(k-1)20}^{20k-1}(D_n^{''}))^{1/6} \\
DA_k^{''} &=(\frac{1}{20}\sum_{n=(k-1)20}^{20k-1}(DA_n^{''}))^{1/6}
\end{align*}
$$
取L2范数
$$
\begin{align*}
d_{sym} &=(\frac{\sum_k(D_k^{'''}t_k)^2}{\sum_k(t_k)^2})^{1/2} \\
d_{asym} &=(\frac{\sum_k(DA_k^{'''}t_k)^2}{\sum_k(t_k)^2})^{1/2}
\end{align*}
$$
其中$D_k^{''}$和$DA_k^{''}$是对称帧损伤和非对称帧损伤，其避免了一些坏区，即忽略了帧损伤值超过阈值的区域。$t_k$是针对帧损伤值的权重，其取决于信号长度。PESQ的最终计算公式如下：
$$
\begin{align*}
PESQ = 4.5-0.1\cdot d_{sym}-0.0309\cdot d_{asym}
\end{align*}
$$
根据ITU-T P.862建议，针对窄带语音，PESQ的值位于-0.5到4.5之间，针对宽带语音，将是一个在接近1到4.64的值。

