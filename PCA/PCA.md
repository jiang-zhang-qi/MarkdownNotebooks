# Principal Component Analysis (PCA)

<!--
姜璋琪
2022年11月5日
-->

<!--Table居中显示-->
<style>
table
{
    margin: auto;
}
</style>

PCA是一种经典的**线性**降维方法.

## 基本想法

线性降维的本质是希望能在高维样本空间$\mathbf{V}$中找到少量的线性无关的向量,使得高维样本能通过这些向量近似表出且尽可能保持原有的“信息量”.这时候就遇到了与SVM相同的问题:

**我应该选择什么样的向量？**

出于不同的目的,我们会得到对向量不同的限制,从而衍生出了很多不同的模型.比如为了让高维样本空间中样本之间的距离在低维空间中得以保持,我们就得到了“多维放缩”(Multiple Dimensional Scaling, MDS)方法.而PCA希望找到的向量能使样本在该向量上的**方差尽可能大**(i.e. 投影尽可能分开,最大方差性),另一种说法是希望样本在该向量上的**投影平均均方误差尽可能小**(i.e. 样本离向量决定的超平面越近越好,最近重构性).

**为什么这样的向量是适合的？**换句话说,是什么原因促使PCA这样选择向量？

回答这个问题之前,我们需要问一下自己为什么需要降维？特征冗余和可视化是两个促使我们发展降维方法的很强的动机.可视化很好理解,是人们希望看一下高维样本的分布等直观特征,但是我们没办法想象3维以上的空间,我们只能选择降到3维及以下的空间去作图.那么特征冗余是什么呢？假设我们有身高和体重这两个特征,在一般情况下这两个特征都是如Figure 1所呈现的近似线性性的(对成年人来说,标准体重=(身高cm-70)x0.6),那么我们说如果我们知道了身高然后去猜体重就不完全是随机的！是可以通过身高近似推断出体重的,也就是这两个特征之间存在高度关联性,两个特征所要表述的信息存在高度重叠,此时我们就称特征存在冗余.这种特征冗余可能会使算法产生额外的开销,甚至对最终模型的表现有负面效果,因此我们希望通过降维方法去消除这些冗余.
<center>
    <img src="https://raw.githubusercontent.com/jiang-zhang-qi/MarkdownNotebooks/main/PCA/Figures/figure1.png" style="zoom:15%" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 1. 身高与体重之间呈现近似线性性</div>
    </br>
</center>
现在回答之前的问题,**PCA选择的向量实际上是在降低特征之间的线性相关性**,Figure 2是PCA降维示意图,红色叉表示原始数据,蓝线表示PCA选择的向量所构成的一维坐标,蓝色叉从蓝色坐标的角度看是原样本的低维投影,从原样本空间(黑色坐标)来看是低维投影的重构向量.我们发现,重构向量距离对应的原始样本很近,这就意味着低维投影能被很好地还原成原始样本(i.e. 低维投影包含了原始样本中绝大部分“信息”),并且我们只需要知道原始向量在蓝色坐标上的一维投影坐标就可以近似表示样本了(i.e. 用一个新特征表示原来高度相关的两个特征),这样从整个数据集的角度看,PCA即保持了原始数据集的“信息”,又降低了原始数据集特征的冗余.
<center>
    <img src="https://raw.githubusercontent.com/jiang-zhang-qi/MarkdownNotebooks/main/PCA/Figures/figure2.png" style="zoom:15%" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 2. PCA在身高体重数据集中选择的向量以及低维投影的示意图</div>
    </br>
</center>

**PCA是如何找到这样的向量的？**

PCA先找到第一个向量$\mathbf{w}_{1}$使得原始数据集在$\mathbf{w}_{1}$上投影的方差最大(或者投影平均均方误差最小),然后在$\mathbf{w}_{1}$的正交补空间中,再找到第二个使得原始数据集投影的方差最大(或者投影平均均方误差最小)的向量$\mathbf{w}_{2}$.重复上述过程直至找到$k$个这样的向量(如果我们希望降到$k$维的话).

## 模型的推导(Model)

现在,我们把基本想法转化为数学描述.我们的目标是*将$d$维样本降维至$k$维*.
<!-- 模糊的哲学与精确的数学——人类的望远镜与显微镜.	(《数学与哲学》张景中) -->

我们先做一些约定:

* 给定$m$个$d$维样本空间$\mathbf{V}$中的样本$\mathbf{x}_{i}(i=1,2,\cdots,m)$并以列向量形式排列成一个样本矩阵$\mathbf{X} = \{\mathbf{x}_{1},\mathbf{x}_{2}, \cdots, \mathbf{x}_{m}\} \in \mathbb{R}^{d \times m}$.

* 假设数据样本已经做了**中心化**(i.e. $\sum_{i=1}^{m}\mathbf{x}_{i} = \mathbf{0}$),这是PCA的“起手式”,保证了我们之后计算协方差矩阵的简便.

* 设我们要找的$k(k < d)$个线性无关的向量为$\mathbf{w}_{1}, \mathbf{w}_{2}, \cdots, \mathbf{w}_{k}$,其中$\mathbf{w}_{i} \in \mathbb{R}^{d \times 1} \ s.t.\ \mathbf{w}_{i}^{T}\mathbf{w}_{j} = \delta_{ij}(i,j=1,2,\cdots,k),$
$$ 
\delta_{ij} = 
\begin{cases}
0, \quad i \neq j, \\
1, \quad i = j.
\end{cases}
$$
将$\mathbf{w}_{1}, \mathbf{w}_{2}, \cdots, \mathbf{w}_{k}$以列向量的形式排列成的一个矩阵记为$\mathbf{W} \in \mathbb{R}^{d \times k}$,由$\mathbf{w}_{i}$的性质可知$\mathbf{W}$满足$\mathbf{W}^{T}\mathbf{W}=\mathbf{I}_{k}$.并记$\mathbf{U}=span\{\mathbf{w}_{1}, \mathbf{w}_{2}, \cdots, \mathbf{w}_{k}\}$.

* 记样本点$\mathbf{x}_{i}$在$\mathbf{U}$中以$\mathbf{w}_{1}, \mathbf{w}_{2}, \cdots, \mathbf{w}_{k}$为基的投影坐标为$\mathbf{z}_{i}=(z_{i,1}, z_{i,2}, \cdots, z_{i,k})^{T}.$

对任意$j=1,2,\cdots,k$有
$$
\begin{align}
\mathbf{w}_{j}^{T}\mathbf{x}_{i} & = (\mathbf{w}_{j}, \mathbf{x}_{i}) \\
& = ||\mathbf{w}_{j}||_{2}||\mathbf{x}_{i}||_{2}cos<\mathbf{w}_{j}, \mathbf{x}_{i}> \\
& = ||\mathbf{x}_{i}||_{2}cos<\mathbf{w}_{j}, \mathbf{x}_{i}>, \tag{1}
\end{align}
$$
其中$(\cdot,\cdot)$表示内积,第三个等号用到了$||\mathbf{w}_{j}||_{2}=1.$式(1)表示输入向量$\mathbf{x}_{i}$在$\mathbf{w}_{j}$上的投影长度,因此有$\mathbf{w}_{j}^{T}\mathbf{x}_{i}=z_{i,j}$.故投影坐标可表示为:
$$
\mathbf{z}_{i} = \mathbf{W}^{T}\mathbf{x}_{i}. \tag{2}
$$
而基于$\mathbf{z}_{i}$对输入向量$\mathbf{x}_{i}$的重构向量$\hat{\mathbf{x}_{i}}$可表示为:
$$
\hat{\mathbf{x}_{i}} = \mathbf{W}\mathbf{z}_{i}. \tag{3}
$$

### 1. 最近重构性
在基本思想中有一个思路是希望低维样本重构后与原高维样本尽可能接近,现在我们将这个思路转化为数学描述.

考虑整个数据集,基于投影坐标$\mathbf{z}_{i}$重构后的向量$\hat{\mathbf{x}_{i}}$与原始高维向量$\mathbf{x}_{i}$之间的距离可以表示为(注:这里除不除$m$都是可以的,对最优化问题的解没有影响):
$$
\sum_{i=1}^{m}||\hat{\mathbf{x}_{i}} - \mathbf{x}_{i}||_{2}^{2}, \tag{4}
$$
再将式(2),(3)带入式(4)得
$$
\sum_{i=1}^{m}||\mathbf{W}\mathbf{W}^{T}\mathbf{x}_{i} - \mathbf{x}_{i}||_{2}^{2},
$$
利用$L_{2}$范数的定义可得
$$
\begin{align}
\sum_{i=1}^{m}(\mathbf{W}\mathbf{W}^{T}\mathbf{x}_{i} - \mathbf{x}_{i})^{T}(\mathbf{W}\mathbf{W}^{T}\mathbf{x}_{i} - \mathbf{x}_{i}) & = \sum_{i=1}^{m} \mathbf{x}_{i}^{T}(\mathbf{W}\mathbf{W}^{T} - \mathbf{I}_{d})^{T}(\mathbf{W}\mathbf{W}^{T} - \mathbf{I}_{d})\mathbf{x}_{i} \\
& = tr(\mathbf{X}^{T}(\mathbf{W}\mathbf{W}^{T} - \mathbf{I}_{d})^{T}(\mathbf{W}\mathbf{W}^{T} - \mathbf{I}_{d})\mathbf{X}) \\
& = tr((\mathbf{W}\mathbf{W}^{T}\mathbf{X} - \mathbf{X})^{T}(\mathbf{W}\mathbf{W}^{T}\mathbf{X} - \mathbf{X})) \\
& = tr(\mathbf{X}^{T}\mathbf{W}\mathbf{W}^{T}\mathbf{W}\mathbf{W}^{T}\mathbf{X} - 2\mathbf{X}^{T}\mathbf{W}\mathbf{W}^{T}\mathbf{X} + \mathbf{X}^{T}\mathbf{X}) \\
& = tr(\mathbf{X}^{T}\mathbf{W}\mathbf{W}^{T}\mathbf{X}) - 2tr(\mathbf{X}^{T}\mathbf{W}\mathbf{W}^{T}\mathbf{X}) + tr(\mathbf{X}^{T}\mathbf{X}) \\
& = -tr(\mathbf{X}^{T}\mathbf{W}\mathbf{W}^{T}\mathbf{X}) + tr(\mathbf{X}^{T}\mathbf{X}) \\
& = -tr(\mathbf{W}^{T}\mathbf{X}\mathbf{X}^{T}\mathbf{W}) + tr(\mathbf{X}^{T}\mathbf{X}). \tag{5}
\end{align}
$$
第五个等号用到了迹的线性性,第七个等号用到了$tr(AB)=tr(BA)$.根据最近重构性,式(5)应该被最小化,则目标函数可写为:
$$
\begin{align}
\min_{\mathbf{W}} \ & -tr(\mathbf{W}^{T}\mathbf{X}\mathbf{X}^{T}\mathbf{W}) + tr(\mathbf{X}^{T}\mathbf{X}) \\
& s.t. \ \mathbf{W}^{T}\mathbf{W} = \mathbf{I}_{k}. \tag{6}
\end{align}
$$
由于$tr(\mathbf{X}^{T}\mathbf{X})$与$\mathbf{W}$无关,因此(6)的最优化问题等价于:
$$
\begin{align}
\min_{\mathbf{W}} \ & -tr(\mathbf{W}^{T}\mathbf{X}\mathbf{X}^{T}\mathbf{W}) \\
& s.t. \ \mathbf{W}^{T}\mathbf{W} = \mathbf{I}_{k}. \tag{7}
\end{align}
$$
### 2. 最大可分(方差)性

现在我们再沿着最大方差性推导,i.e. 投影坐标具有最大的方差.

我们从样本的投影坐标的方差定义出发:
$$
\frac{1}{m}\sum_{i=1}^{m}(\mathbf{z}_{i} - \frac{1}{m}\sum_{j=1}^{m} \mathbf{z}_{j})^{T}(\mathbf{z}_{i} - \frac{1}{m}\sum_{i=j}^{m} \mathbf{z}_{j}), \tag{8}
$$
幸运的是我们的数据已经经过中心化了(i.e. $\sum_{j=1}^{m}\mathbf{z}_{j} = \mathbf{0}$),因此式(7)可以改写为:
$$
\begin{align}
\frac{1}{m}\sum_{i=1}^{m}\mathbf{z}_{i}^{T}\mathbf{z}_{i} & = \frac{1}{m} \sum_{i=1}^{m} (\mathbf{W}^{T}\mathbf{x}_{i})^{T}(\mathbf{W}^{T}\mathbf{x}_{i}) \\
& = \frac{1}{m} \sum_{i=1}^{m} \mathbf{x}_{i}^{T}\mathbf{W}\mathbf{W}^{T}\mathbf{x}_{i} \\
& = \frac{1}{m} tr(\mathbf{X}^{T}\mathbf{W}\mathbf{W}^{T}\mathbf{X}) \\
& = \frac{1}{m} tr(\mathbf{W}^{T}\mathbf{X}\mathbf{X}^{T}\mathbf{W}). \tag{9}
\end{align}
$$
第一个等号用到了式(2).根据最大方差性,式(8)应该被最大化,则目标函数可写为:
$$
\begin{align}
\max_{\mathbf{W}} \ & tr(\mathbf{W}^{T}\mathbf{X}\mathbf{X}^{T}\mathbf{W}) \\
& s.t. \ \mathbf{W}^{T}\mathbf{W} = \mathbf{I}_{k}, \tag{10}
\end{align}
$$
其中$1/m$被省略了,因为最优化问题乘上一个常数对于最优参数$\mathbf{W}$的选择没有影响.

### 3. 这两种思路是殊途同归的！

我们发现根据最近重构性和最大方差性两种思路推导出来的目标函数(7)和(10)是一致的.我们现在来看一看为什么会这样？
将(6)式中的目标函数写成和式的形式:
$$
-tr(\mathbf{Z}^{T}\mathbf{Z}) + tr(\mathbf{X}^{T}\mathbf{X}) = \sum_{i=1}^{m} (\mathbf{x}_{i}^{T}\mathbf{x}_{i} - \mathbf{z}_{i}^{T}\mathbf{z}_{i}),
$$
对于单个样本来说$\mathbf{x}_{i}^{T}\mathbf{x}_{i}$是$\mathbf{x}_{i}$的膜长平方,$\mathbf{z}_{i}^{T}\mathbf{z}_{i}$是$\mathbf{x}_{i}$在低维子空间$\mathbf{U}$上投影的膜长平方,如Figure 3的示意图所示.
<center>
    <img src="https://raw.githubusercontent.com/jiang-zhang-qi/MarkdownNotebooks/main/PCA/Figures/figure3.png" style="zoom:25%" />
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig. 3. 最大方差性和最近重构性直观示意图</div>
    </br>
</center>
由于$\mathbf{z}_{i}$是$\mathbf{x}_{i}$在$\mathbf{U}$上的投影,因此有勾股定理成立(当然也可以认为是加一项减一项):
$$
\mathbf{x}_{i}^{T}\mathbf{x}_{i} = \underbrace{\mathbf{z}_{i}^{T}\mathbf{z}_{i}}_{方差} +\underbrace{(\mathbf{x}_{i}^{T}\mathbf{x}_{i} - \mathbf{z}_{i}^{T}\mathbf{z}_{i})}_{投影误差}
$$
由于等号左边是一个固定的数,因此最近重构性去最小化投影误差就等价于最大方差性去最大化投影膜长.因此这两种思路是殊途同归的！

## 向量求解算法的推导(Algorithm)

我们现在依据最大方差性,按照PCA依次找向量的算法去推导$\mathbf{w}_{i}$的表达式.记对称阵$\mathbf{X}\mathbf{X}^{T} = \sum$,由于实对称阵必可对角化,不妨设$\sum$的特征值从大到小排列为$\lambda_{1} \geq \lambda_{2} \geq \cdots \geq \lambda_{d}$(重根分开写了).

**第1个向量$\mathbf{w}_{1}$**

由在$\mathbf{w}_{1}$上的投影方差最大原则,有
$$
\begin{align}
\max_{\mathbf{w}_{1}}\ \ & \mathbf{w}_{1}^{T}\sum\mathbf{w}_{1} \\
& s.t. \ \mathbf{w}_{1}^{T}\mathbf{w}_{1} = 1. \tag{11} 
\end{align}
$$
由Lagrange乘子法,构造Lagrange函数有
$$
\mathcal{L}(\mathbf{w}_{1}, \alpha) = \mathbf{w}_{1}^{T}\sum\mathbf{w}_{1} + \alpha(1 - \mathbf{w}_{1}^{T}\mathbf{w}_{1}). \tag{12}
$$
分别对$\mathbf{w}_{1}, \alpha$求偏导并使偏导为$0$得,
$$
\begin{align}
& \frac{\partial \mathcal{L}(\mathbf{w}_{1}, \alpha)}{\partial \mathbf{w}_{1}} = 0 \Rightarrow \quad \sum\mathbf{w}_{1} = \alpha\mathbf{w}_{1},  \tag{13}\\
& \frac{\partial \mathcal{L}(\mathbf{w}_{1}, \alpha)}{\partial \alpha} = 0 \Rightarrow \quad 1 = \mathbf{w}_{1}^{T}\mathbf{w}_{1}, \tag{14}
\end{align}
$$
式(13)用到了$\sum^{T} = \sum,\ \frac{\partial \mathbf{w}_{1}^{T}\sum\mathbf{w}_{1}}{\partial \mathbf{w}_{1}} = (\sum^{T} + \sum)\mathbf{w}_{1} = 2\sum\mathbf{w}_{1},\ \frac{\partial \mathbf{w}_{1}^{T}\mathbf{w}_{1}}{\partial \mathbf{w}_{1}} = 2\mathbf{w}_{1}.$
由式(13)可知我们要求的$\alpha, \mathbf{w}_{1}$分别是对称阵$\sum$的特征值及其对应的特征向量,式(14)告诉我们该特征向量必须是单位向量.但是我们应该怎么选择特征向量呢？我们观察式(13),对其两边同时乘上$\mathbf{w}_{1}^{T}$有
$$
\mathbf{w}_{1}^{T}\sum\mathbf{w}_{1} = \alpha\mathbf{w}_{1}^{T}\mathbf{w}_{1} = \alpha, \tag{15}
$$
我们会发现我们的目标函数就是$\sum$的特征值$\alpha$,那么为了最大化式(15)就等价于让$\alpha$取$\lambda_{1}$,则$\mathbf{w}_{1}$就可以取为$\lambda_{1}$对应的单位特征向量.

**第2个向量$\mathbf{w}_{2}$**

第二个向量$\mathbf{w}_{2}$是在$\mathbf{w}_{1}$的正交补空间中寻找的,相比于优化问题(11)多了一个限制$\mathbf{w}_{2}^{T}\mathbf{w}_{1}=0$.于是,为了求解$\mathbf{w}_{2}$,我们就需要求解下面的优化问题:
$$
\begin{align}
\max_{\mathbf{w}_{2}} \ & \mathbf{w}_{2}^{T}\sum\mathbf{w}_{2} \\
& s.t. \ \mathbf{w}_{2}^{T}\mathbf{w}_{2} = 1 \\
& \quad \ \ \mathbf{w}_{2}^{T}\mathbf{w}_{1} = 0. \tag{16}
\end{align}
$$
仍然利用Lagrange乘子法,构造Lagrange函数有
$$
\mathcal{L}(\mathbf{w}_{2}, \alpha, \beta) = \mathbf{w}_{2}^{T}\sum\mathbf{w}_{2} + \alpha(1 - \mathbf{w}_{2}^{T}\mathbf{w}_{2}) + \beta\mathbf{w}_{2}^{T}\mathbf{w}_{1}. \tag{17}
$$
对$\mathbf{w}_{2}$求偏导并使偏导为$0$得,
$$
\begin{align}
& \frac{\partial \mathcal{L}(\mathbf{w}_{2}, \alpha, \beta)}{\partial \mathbf{w}_{2}} = 0 \Rightarrow \ 2\sum\mathbf{w}_{2} - 2\alpha\mathbf{w}_{2} + \beta\mathbf{w}_{1} = 0. \tag{18}
\end{align}
$$
对(18)式两边左乘$\mathbf{w}_{1}^{T}$有
$$
2\mathbf{w}_{1}^{T}\sum\mathbf{w}_{2} - 2\alpha\mathbf{w}_{1}^{T}\mathbf{w}_{2} + \beta\mathbf{w}_{1}^{T}\mathbf{w}_{1} = 0,
$$
由于
$$
\mathbf{w}_{1}^{T}\sum\mathbf{w}_{2} = (\mathbf{w}_{1}^{T}\sum\mathbf{w}_{2})^{T} = \mathbf{w}_{2}^{T}\sum\mathbf{w}_{1} = \lambda_{1}\mathbf{w}_{2}^{T}\mathbf{w}_{1} = 0,
$$
且$\mathbf{w}_{1}^{T}\mathbf{w}_{1} = 1$,我们有
$$
\beta = 0.
$$

那么(18)式可进一步化简为:
$$
2\sum\mathbf{w}_{2} - 2\alpha\mathbf{w}_{2} = 0 \Rightarrow \quad \sum\mathbf{w}_{2} = \alpha\mathbf{w}_{2}. \tag{19}
$$
对(19)式两边左乘$\mathbf{w}_{2}^{T}$,可将目标函数化简为
$$
\mathbf{w}_{2}^{T}\sum\mathbf{w}_{2} = \alpha\mathbf{w}_{2}^{T}\mathbf{w}_{2} = \alpha.
$$
上述过程启发了我们:$\alpha, \mathbf{w}_{2}$分别是对称阵$\sum$的特征值及其对应的特征向量;$\mathbf{w}_{2}$要是单位向量且与$\mathbf{w}_{1}$正交.那么为了最大化(16)式(i.e.最大化$\alpha$),我们就取$\alpha=\lambda_{2}$, $\mathbf{w}_{2}$为$\lambda_{2}$对应的单位特征向量(若在$\lambda_{2} < \lambda_{1}$的条件下让$\alpha=\lambda_{1}$,则找不到满足要求的单位特征向量$\mathbf{w}_{2}$).

> 这里有个地方需要注意一下:
> 
> 我们前面假设$\sum$的特征值从大到小排列为$\lambda_{1} \geq \lambda_{2} \geq \cdots \geq \lambda_{d}$,是把重根分开来写的,也就是说此时$\lambda_{2}$有两种情况
> 
> $1^{o} \lambda_{2} = \lambda_{1}$:此时$\mathbf{w}_{1}$与$\mathbf{w}_{2}$属于同一个特征子空间,而$\mathbf{w}_{1}$是已经选定的一个单位向量.那么$\mathbf{w}_{2}$只要在$\lambda_{1}$的特征子空间里先选择与$\mathbf{w}_{1}$线性无关的向量,然后通过Schmid正交化,最后单位化就能得到满足条件$\mathbf{w}_{1}^{T}\mathbf{w}_{2}=0, \mathbf{w}_{2}^{T}\mathbf{w}_{2}=1$的单位特征向量$\mathbf{w}_{2}$.
> 
> $2^{o} \lambda_{2} < \lambda_{1}$:此时$\mathbf{w}_{1}$与$\mathbf{w}_{2}$分属不同的特征子空间,由属于不同特征子空间的向量必相互正交可知,只要在$\lambda_{2}$特征子空间里找一个单位向量就能得到满足条件的$\mathbf{w}_{2}$.

**第$j$个向量$\mathbf{w}_{j}$**

由数学归纳法容易证明,取$\mathbf{w}_{j}$为特征值$\lambda_{j}$对应的单位正交特征向量,此时$\mathbf{w}_{j}^{T}\sum\mathbf{w}_{j}$最大为$\lambda_{j}$.其中$\mathbf{w}_{j}$在找的时候同样需要注意上面Block里的情况.


总结一下,我们要去求解$\mathbf{w}_{1}, \mathbf{w}_{2}, \cdots, \mathbf{w}_{k}$,我们只需要对$\sum$做特征值分解,并取最大的前$k$个特征值所对应的特征向量,再通过schmid正交化、单位化等手段,化成满足条件的$\mathbf{w}_{1}, \mathbf{w}_{2}, \cdots, \mathbf{w}_{k}$即可.我们在实际运用过程中,我们常用Singular Value Decompensation (SVD)技术直接求得正交化、单位化后的向量.

当然,如果我们使用SVD去找PCA的主成分向量的话,我们可以直接对样本矩阵$\mathbf{X}$进行SVD分解.

设$\mathbf{X}$能被分解成下述形式:
$$
\mathbf{X} = \mathbf{U} \mathbf{S} \mathbf{V}^{T},
$$
其中$\mathbf{U}$为$d \times d$的酉矩阵(每列均为$\mathbf{X}$的左奇异向量),$\mathbf{V}^{T}$为$m \times m$的酉矩阵(每行均为$\mathbf{X}$的右奇异向量),$\mathbf{S}$为$d \times m$的矩阵,且“主对角线”位置上的元素为$\mathbf{X}$的奇异值(一般是从大到小排列的$\sigma_{1} \geq \sigma_{2} \geq \cdots \geq \sigma_{min\{m, d\}}$).那么
$$
\begin{align}
\mathbf{X}\mathbf{X}^{T} & = \mathbf{U} \mathbf{S} \mathbf{V}^{T} \mathbf{V} \mathbf{S}^{T} \mathbf{U}^{T} \\
& = \mathbf{U} \mathbf{S} \mathbf{S}^{T} \mathbf{U}^{T}.
\end{align}
$$
此时,如果我们取$\mathbf{w}_{1}, \mathbf{w}_{2}, \cdots, \mathbf{w}_{k}$为$\mathbf{U}$的前$k$列所对应的向量,则
$$
\begin{align}
tr(\mathbf{W}^{T}\mathbf{X}\mathbf{X}^{T}\mathbf{W}) & = tr(\mathbf{W}^{T}\mathbf{U} \mathbf{S} \mathbf{S}^{T} \mathbf{U}^{T}\mathbf{W}) \\
& = \sum_{i=1}^{k} \sigma_{i}^{2} \\
& = \sum_{i=1}^{k} \lambda_{i},
\end{align}
$$
其中第三个等号用到了$\sigma_{i}^{2} = \lambda_{i}$($\lambda_{i}$的定义与前文中的一样是$\mathbf{X}\mathbf{X}^{T}$的第$i$大特征值).我们发现这样取出来的向量,与我们前述分析的最优化问题的最值是相同的,都是$\sum_{i=1}^{k} \lambda_{i}$,因此我们也常这样去找PCA的主成分向量.

## 应该选择几个向量？(Strategy)

当使用PCA对数据进行降维的时候,我们需要给定一个值$k$(i.e. 降维后的维数要给定).$k$的选取往往是偏主观的,因为如果要可视化,我们选取的维数不得不是$2$或$3$.但是我们还是希望能有一个评估手段,告诉我们如果我选取某个$k$值,那么这个降维结果是好是坏？

现在,从信息压缩的角度去制定一个评价准则:
$$
\frac{平均损失的信息量}{平均原始信息量} = \frac{\frac{1}{m}\sum_{i=1}^{m} ||\mathbf{x}_{i} - \hat{\mathbf{x}_{i}}||_{2}^{2}}{\frac{1}{m}\sum_{i=1}^{m} ||\mathbf{x}_{i}||_{2}^{2}}. \tag{20}
$$
现在我们进行化简(20)式:
$$
\begin{align}
(20) & = \frac{tr((\mathbf{X} - \mathbf{W}\mathbf{W}^{T}\mathbf{X})^{T}(\mathbf{X} - \mathbf{W}\mathbf{W}^{T}\mathbf{X}))}{tr(\mathbf{X}^{T}\mathbf{X})} \\
& = \frac{tr(\mathbf{X}^{T}\mathbf{X} - \mathbf{X}^{T}\mathbf{W}\mathbf{W}^{T}\mathbf{X} - \mathbf{X}^{T}\mathbf{W}\mathbf{W}^{T}\mathbf{X} + \mathbf{X}^{T}\mathbf{W}\mathbf{W}^{T}\mathbf{W}\mathbf{W}^{T}\mathbf{X})}{tr(\mathbf{X}^{T}\mathbf{X})} \\
& = \frac{tr(\mathbf{X}^{T}\mathbf{X} - \mathbf{X}^{T}\mathbf{W}\mathbf{W}^{T}\mathbf{X})}{tr(\mathbf{X}^{T}\mathbf{X})} \\
& = 1 - \frac{tr(\mathbf{X}^{T}\mathbf{W}\mathbf{W}^{T}\mathbf{X})}{tr(\mathbf{X}^{T}\mathbf{X})}. \tag{21}
\end{align}
$$
其中第三个等号用到了$\mathbf{W}^{T}\mathbf{W} = \mathbf{I}_{k}$.由于$\mathbf{W}$是取了最大的前$k$个特征值对应的特征向量,现在把后面的$d-k$个经过Schmid正交化和单位化的单位正交特征向量补上,构成$\widetilde{\mathbf{W}} = (\mathbf{W}, \mathbf{w}_{k+1}, \cdots, \mathbf{w}_{d}) \in \mathbb{R}^{d \times d}\ s.t.\ \widetilde{\mathbf{W}}^{T}\widetilde{\mathbf{W}} = \mathbf{I}_{d}.$于是(21)可被改写为:
$$
\begin{align}
(21) & = 1 - \frac{tr(\mathbf{W}^{T}\mathbf{X}\mathbf{X}^{T}\mathbf{W})}{tr(\mathbf{X}^{T}\widetilde{\mathbf{W}}\widetilde{\mathbf{W}}^{T}\mathbf{X})} \\
& = 1 - \frac{tr(\mathbf{W}^{T}\mathbf{X}\mathbf{X}^{T}\mathbf{W})}{tr(\widetilde{\mathbf{W}}^{T}\mathbf{X}\mathbf{X}^{T}\widetilde{\mathbf{W}})} \\
& = 1 - \frac{\sum_{i=1}^{k}\lambda_{i}}{\sum_{i=1}^{d}\lambda_{i}}, \tag{22}
\end{align}
$$
其中第三个等号是利用了特征值与特征向量的关系.我们说PCA降维的表现好或者压缩损失小,如果(22)式小于等于0.01(或0.05),i.e.
$$
1 - \frac{\sum_{i=1}^{k}\lambda_{i}}{\sum_{i=1}^{d}\lambda_{i}} \leq 0.01(0.05). \tag{23}
$$
换句话说就是PCA降维后,低维的投影坐标保持了原有数据集99%(95%)的信息量,i.e.
$$
\frac{\sum_{i=1}^{k}\lambda_{i}}{\sum_{i=1}^{d}\lambda_{i}} \geq 0.99(0.95). \tag{24}
$$
于是我们就可以用不等式(23)或(24)去判断我们的$k$值选取得恰不恰当了！
## PCA的伪代码(Pseudo-code)

$$
\begin{align}
Input:& \hspace{100cm}\\
& 样本集合\mathcal{D} = \{\mathbf{x}_{1}, \mathbf{x}_{2}, \cdots, \mathbf{x}_{m}\}; \\
& 低维空间的维数k.
\end{align}
$$

$$
\begin{align}
Process:& \hspace{100cm}\\
& 1.对所有样本进行中心化:\mathbf{x}_{i} := \mathbf{x}_{i} - \frac{1}{m}\sum_{i=1}^{m}\mathbf{x}_{i}; \\
& 2.计算样本的协方差矩阵\ \mathbf{X}^{T}\mathbf{X}; \\
& 3.对协方差矩阵\mathbf{X}^{T}\mathbf{X}做特征值分解; \\
& 4.取最大的前k个特征值所对应的特征向量\mathbf{w}_{1}, \mathbf{w}_{2}, \cdots, \mathbf{w}_{k}.
\end{align}
$$

$$
\begin{align}
Output: \ \mathbf{W} = (\mathbf{w}_{1}, \mathbf{w}_{2}, \cdots, \mathbf{w}_{k}). \hspace{100cm}
\end{align}
$$


## Reference
[1] 周志华.机器学习[M].北京:清华大学出版社,2016:229-232.

[2] 李航.统计学习方法[M].北京:清华大学出版社,2019:301-303.


