# FingerprintFeatureExtractionAndDescription
Fingerprint feature extraction and description 指纹特征提取及描述

看不到图片请移步[我的这篇博客](https://juejin.cn/post/6924305350497828872/)，转载注明出处

记得**star一下**呀

## 最终效果图

![最终效果图](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/fdb21a32bcb342909bb1889256fd83b3~tplv-k3u1fbpfcp-watermark.image)

## 设计思路

项目的编程环境为python3.7.7，编译器使用pycharm2019.3.4 x64。首先为项目设计一个GUI界面，界面有四个按钮，分别是“选择图片”、“图像增强”、“细化”和“特征提取及描述”，使用是按顺序点击按钮即可，每完成一步，都会在按钮下方显示处理结果，最终的特征描述会在右下的文本框中显示。

### 图像增强

- **图像归一化**

由于不同的指纹图像在灰度图分布上有很大的差异，会给之后的图像特征提取和匹配增加难度，因此指纹图像要进行归一化处理，将所有图像转换成具有给定均值和方差的标准图像。归一化并不能改变指纹的脊线和谷线的清晰程度，其结果是减少了不同指纹图像之间灰度值的差异，并为接下来的图像处理做好准备。归一化公式如下：

$$
G_{\left(i,j\right)}=\left\{
\begin{aligned}
&M_0+\sqrt{\frac{\sigma_0^2(I(i,j)-M)^2}{\sigma^2}} &I(i,j)>M\\
&M_0-\sqrt{\frac{\sigma_0^2(I(i,j)-M)^2}{\sigma^2}} &I(i,j)\le M
\end{aligned}
\right.
$$

式中$I\left(i,j\right)$和$G_{\left(i,j\right)}$分别为规格化前后的图像，$M_0$、$\sigma_0^2$是预先设定的图像平均灰度和均方差，$M$和$\sigma^2$为原图像的灰度均值和方差。

- **方向场估算**

方向场反映了指纹图像上纹线的方向，其准确性直接影响图像增强的效果。根据纹线方向在局部区域内基本一致的特点，先把指纹图像分块，然后计算每一个子块的纹线方向，最后用该方向代表对应子块内各个像素的方向。用这种方法来求指纹方向场效率较高且不易受少量的图像噪声影响，具体算法如下：

1. 把归一化图像分成$16\times16$的像素块，然后利用Sobel算子计算块中每个像素点水平方向上的梯度值$\partial x\left(u,\nu\right)$和垂直方向上的梯度值$\partial y\left(u,v\right)$。

2. 计算中心点为$\left(i,j\right)$的边长为w的子块的脊线方向角$\theta\left(i,j\right)$，公式如下：

$$
V_x\left(i,j\right)=\sum_{u=i-\frac{w}{2}}^{i+\frac{w}{2}}\sum_{v=i-\frac{w}{2}}^{i+\frac{w}{2}}2\partial x\left(u,\nu\right)\partial y\left(u,v\right)
$$

$$
V_y\left(i,j\right)=\sum_{u=i-\frac{w}{2}}^{i+\frac{w}{2}}\sum_{v=i-\frac{w}{2}}^{i+\frac{w}{2}}{\partial x^2\left(u,\nu\right)\partial y^2\left(u,v\right)}
$$

$$
\theta\left(i,j\right)=\frac{1}{2}\tan^{-1}{\left(\frac{v_y\left(i,j\right)}{v_x\left(i,j\right)}\right)}
$$

<div align="center">
    <img src="https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6599400e53e3466c91c8cb60eaadb34f~tplv-k3u1fbpfcp-watermark.image" height = "180" alt=""/>
    <br>
    <p>指纹图像的方向场</p>
</div>

- **频率场估算**

在指纹图像的局部非奇异区域里，沿着垂直于脊线的方向看，指纹脊线和谷线像素点灰度值大致形成一个二维的正弦波，定义纹线频率为相邻的两个波峰或波谷之间的像素点数的倒数。求取这些互不重叠的局部区域的频率值，按各区域位置组成一个场结构，称为指纹的频率场。设$N$表示规格化后的图像，$O$是指纹方向场，算法如下：

在所得方向场的基础上，沿其垂直方向投影每一块所有像素的灰度值，该投影形成一维正弦波，其极值点对应指纹的脊线和谷线。假定$T\left(i,j\right)$作为上述一维正弦波两相邻峰值之间的平均像素数，则频率$F\left(i,j\right)=\frac{1}{T\left(i,j\right)}$。

- **gabor滤波**

一旦指纹图像的方向场和频率场确定，这些参数可以用来构造偶对称Gabor滤波器。Gabor滤波器是具有方向选择特性和频率选择特性的带通滤波器，并且能够达到时域和频域的最佳结合。偶对称的Gabor滤波器在空间域中具有下面的形式：

$$
G\left(x,y,\theta,f\right)=\exp{\left\{-\frac{1}{2}\left(\frac{x_\theta^2}{\sigma_x^2}+\frac{y_\theta^2}{\sigma y^2}\right)\right\}}\cos{\left(2\pi f x_\theta\right)}
$$

$$
x_\theta=xcos\theta+ysin\theta
$$

$$
x_\theta=-xcos\theta+ysin\theta
$$

式中，$\theta$是滤波器的方向，$f$是脊线的频率，$\left[x_\theta,y_\theta\right]$表示坐标轴$\left[x,y\right]$逆时针旋转角度$\theta$，$\sigma_x$，和$\sigma_y$，分别是沿着x和y轴的高斯包络常量。

<div align="center">
    <img src="https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c4a233620e1e442cbc73cc687f6d6a19~tplv-k3u1fbpfcp-watermark.image" height = "250" alt=""/>
    <br>
    <p>原图和图像增强后的效果图</p>
</div>

## 骨架提取

图像骨架提取，实际上就是提取目标在图像上的中心像素轮廓，以目标中心为准，对目标进行细化。一般细化后的目标都是单层像素宽度。细化是从原来的图中去掉一些点，但仍要保持原来的形状。实际上是保持原图的骨架。判断一个点P是否能去掉是以8个相邻点的情况来作为判据的，具体判据为：

- **内部点不能删除**
- **孤立点不能删除**
- **端点不能删除**

如果P是边界点，去掉P后，如果连通分量不增加，则P可删除
<div align="center">
    <img src="https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/db1af33773874ad588729b3439bf77ce~tplv-k3u1fbpfcp-watermark.image" height = "200" alt=""/>
    <br>
    <p>骨架提取</p>
</div>

## 指纹特征提取和表示

本次项目选取的特征是指纹图像的端点和分叉点，特征描述为特征点所处的位置和脊线在特征点处的切线斜率，其中，位置为特征点在图像中的横坐标和纵坐标，端点的切线斜率有1个，分叉点的切线斜率有3个。

- **端点**

若骨架提取后的二值图像的点$\left(i,j\right)$为黑色像素，且其八领域有且只有一个黑色像素，其余7个为白色像素，则点$\left(i,j\right)$为端点，端点特征的提取和表示算法如下：

1. 按上述定义遍历骨架提取后的二值图像的像素点，寻找所有符合定义的端点；
    
2. 剔除指纹图像边缘的端点，因为采集的指纹存在边缘，边缘的端点不能视为指纹的端点，需要予以剔除。方法是看步骤1得到的端点所在的行和列的某一侧是否全为白色像素，是的话判断为边缘，予以剔除，否则判断为内部端点，予以保留；

3. 沿着端点$\left(i,j\right)$所在的脊线，寻找距离端点间隔4个像素点的点$\left(u,v\right)$，若5步之内脊线断裂或遇到分叉点，则将该端点$\left(i,j\right)$剔除，该步骤用于剔除孤立点，并为计算端点的切线斜率做准备；

4. 由步骤3得到的端点$\left(i,j\right)$和间隔4个像素点的点$\left(u,v\right)$，计算脊线的切线斜率，公式如下：

$$
\theta\left(i,j\right)=\left\{
\begin{aligned}
&\frac{\pi}{2} & i&=u,j>v\\
&-\frac{\pi}{2} & i&=u,j<v\\
&tan^{-1}(\frac{i-u}{v-j}) & i&\ne u
\end{aligned}
\right.
$$

- **分叉点**

若骨架提取后的二值图像的点$\left(i,j\right)$为黑色像素，且其八领域有3个黑色像素，其余5个为白色像素，则点$\left(i,j\right)$为端点，端点特征的提取和表示算法如下：

1. 按上述定义遍历骨架提取后的二值图像的像素点，寻找所有符合定义的分叉点；

2. 按照步骤1有些分叉点在像素级别上能得到2个相互贴近的点，需要剔除其中一个。如我们得到下图2个分叉点，在宏观上其实是同一个。项目中剔除原则是邻域黑像素相互贴近的分叉点，所以下图剔除掉第一个，保留第二个；

<div align="center">
    <img src="https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/17cfe06e85be4b5bb511024e67a37900~tplv-k3u1fbpfcp-watermark.image" alt=""/>
    <img src="https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/724c0036c81f4ae0be1de479f35f0337~tplv-k3u1fbpfcp-watermark.image" alt=""/>
    <br>
    <p>像素级别上相互贴近的两个分叉点</p>
</div>

3. 沿着端点$\left(i,j\right)$所在的3条脊线，分别寻找距离端点间隔4个像素点的3个点$\left(u,v\right)$，若5步之内脊线断裂或遇到分叉点，则将该端点$\left(i,j\right)$剔除，为计算端点的切线斜率做准备；

4. 由步骤3得到的端点$\left(i,j\right)$和间隔4个像素点的3个点$\left(u,v\right)$，计算分叉点的3条脊线的切线斜率，公式如下：

$$
\theta\left(i,j\right)=\left\{
\begin{aligned}
&\frac{\pi}{2} & i&=u,j>v\\
&-\frac{\pi}{2} & i&=u,j<v\\
&tan^{-1}(\frac{i-u}{v-j}) & i&\ne u
\end{aligned}
\right.
$$

<div align="center">
    <img src="https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/92d1f8fa61af47179673730217e110d8~tplv-k3u1fbpfcp-watermark.image" alt=""/>
    <br>
    <p>指纹图像的特征提取及描述</p>
</div>

## 参考

[python 简单图像处理（16） 图像的细化（骨架抽取）](https://www.cnblogs.com/xianglan/archive/2011/01/01/1923779.html)

[基于Gabor滤波器的指纹图像增强](https://www.ixueshu.com/document/e1cd035a556029b6.html)

[指纹增强算法的研究](https://www.doc88.com/p-9572139478381.html)

[指纹识别源代码（2）-特征点提取](https://blog.csdn.net/MrCharles/article/details/79300671?ops_request_misc=&request_id=&biz_id=102&utm_source=distribute.pc_search_result.none-task-blog-2~blog~sobaiduweb~default-0)