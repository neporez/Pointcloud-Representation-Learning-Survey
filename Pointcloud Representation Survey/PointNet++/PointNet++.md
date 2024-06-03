
# PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). Pointnet++: Deep hierarchical feature learning on point sets in a metric space. _Advances in neural information processing systems_, _30_.

----
## Abstract

pointcloud를 Deep Learning으로 이전 연구가 몇몇 있었다. PointNet은 이 연구에서 선구적인 연구였지만 PointNet은 local structure의 feature를 담을 수 없었고, 그 결과 섬세하거나 복잡한 패턴을 인지하는 능력이 부족했다. 본 논문에서는 계층적 Neural Network를 PointNet에 적용하여 Local structure의 feature를 담고자 하였다. 또한, point cloud는 일반적으로 다양한 density를 가지도록 sampling되는데, 이는 성능에 악영향을 미치는 것을 확인하였기에, 본 논문에서는 다양한 scale에서 오는 feature를 적용한 새로운 Point Set learning layer를 제안한다. 

----
## Introduction

3D LiDAR의 Pointcloud 특징 설명 -> 생략

PointNet에 대한 설명 -> PointNet google docs 참고

PointNet은 local structure의 feature를 담지 못한다. 하지만, Local structure의 feature는 CNN의 성공에 중요한 요소임이 증명되었다. 

CNN은 입력으로 regular grid(Image, Voxel grid)를 받고, 계층적으로 다양한 해상도의 feature를 담아낼 수 있다. lower lever의 neuron는 작은 receptive field를 가지지만 higher level은 큰 receptive field를 가진다. 계층적으로 local pattern을 축약하는 CNN의 능력은 기존 train set에서 보지 못했던 입력의 경우에도 잘 처리하는 일반화이다.

PointNet++의 아이디어는 간단하다.

1. 공간상에서 일정 distance로 묶인 local region으로 point set을 분배한다. (이때 region은 overlapping될 수 있다.)

2. CNN과 유사하게, 근처 이웃 포인트들로부터 geometric structure의 local feature를 뽑아낸다.

3. 이러한 local feature는 더욱 큰 단위로 묶이게 되고 high level feature로 처리된다.

4. 이 프로세스는 모든 point의 feature를 얻어낼때까지 반복한다.

PointNet++의 주된 이슈는 두가지이다.

1. 어떻게 pointcloud의 분배를 할 것인지

2. 어떻게 pointcloud의 local feature를 학습할 것인지

두가지 이슈는 서로 연관이 있는데, 분배된 point set는 local feature의 가중치가 서로 공유될 수 있도록 분배된 point set끼리 공통된 구조를 만들어야하기 때문이다. 여기서 local feature를 학습하는 방식으로 본 논문에서는 PointNet을 사용했다. PointNet++은 PointNet을 입력 데이터에 대해서 중첩된 분할 point set에 대해서 재귀적으로 적용한 것이다.(파티션에 PointNet을 각각 적용하여 파티션의 Global Feature == Pointcloud의 Local Feature를 뽑아낸다는 뜻)

각각의 파티션(분배된 point set)은 유클리디안 거리의 이웃한 point끼리 결정된다. 전체 pointcloud를 파티션으로 분할하기 위해서 FPS(Farthest Point Sampling) 기법을 사용하였다. 고정된 3D convolution과 비교해서 이 기법은 입력 데이터에 receptive field가 결정되기 때문에 더 효율적이고 효과적이다.

![Pasted image 20240603101707](https://github.com/neporez/Pointcloud-Representation-Learning-Survey/assets/88701811/61bd3bb2-7f50-495f-a5b9-36a02450da9b)


그러나, 이웃 점들 간의 거리를 적절히 조절하는 것은 pointcloud의 non-uniformity에 의해서 아직 풀리지 않은 문제이다. 따라서, 본 논문에서는 pointcloud를 Figure 1의 정도의 density를 가진다고 가정하였다.(Velodyne LiDAR와는 다른 느낌인 듯)

이런 상황에서 sampling 부족으로 파티션에 포인트의 개수가 너무 적어지면 PointNet은 Local Feature를 충분히 뽑아내지 못할 수 있다.

PointNet++의 주요 Contribution은 이러한 문제를 해결하기 위해서 다양한 scale의 neighborhoods를 적용하여 Local Feature를 뽑아내기에 충분한 Robustness를 얻었다는 것이다. PointNet++은 학습 도중에 input set에 대하여 random dropout을 적용하여 네트워크가 다양한 scale의 패턴에 적응할 수 있도록 하였다는 것이다. 

---
## Problem Statement

$\chi = (M,d)$ 는 유클리디안 공간에서 정의된 discrete metric 공간이고, $M \subseteq \mathbb R^n$ 이고, d는 metric의 distance이다. 추가적으로 $M$의 density는 모든 공간에서 일정하지 않다.

본 논문에서는 이러한 입력 $\chi$를 받아 classification or 각 $M$에 대해서 segmentation할 수 있는 set function $f$를 찾는 것이 주 목적이다.

---
## Method

본 논문은 PointNet에 계층 구조를 추가한 확장된 버전의 PointNet으로 볼 수 있다.

PointNet 구조 설명 생략 -> PointNet google docs 참고

#### Hierarchical Point Set Feature Learning

![Pasted image 20240603110523](https://github.com/neporez/Pointcloud-Representation-Learning-Survey/assets/88701811/416e32b0-5612-4fff-bcaf-867ef632031b)

single max pooling을 사용하여 전체 pointcloud를 결합했던 것과는 달리, PointNet++은 point들을 그룹화하고 계층적으로 점점 더 커지는 Region을 갖도록 나아간다.

계층적인 구조는 몇 개의 set abstraction level로 구성되어있다.

set abstraction level은 3가지 구성 요소로 이루어져 있다.

1. Sampling Layer

2. Grouping Layer

3. PointNet Layer

##### Samping Layer

Pointcloud $\{x_1, x_2, ..., x_n\}$ 이 주어지면, FPS를 통해 pointcloud의 subset $\{x_{i_1}, x_{i_2}, ..., x_{i_m}\}$ 를 생성한다. Random Sampling과 비교하여 같은 수의 centroid일 때 pointcloud를 좀 더 많이 담아내는 것을 확인하였다. 

##### Grouping Layer

Input Size는 $N \times (d+C)$이고 $N$은 Point의 개수 $d$는 local region coordination($x_i^{(j)} = x_i^{(j)} - \hat{x}^{(j)}$), $C$는 Feature Channel이다.
Output Size는 $N' \times K \times (d+C)$이고 $N'$은 sampling된 centroid의 개수, $K$는 각 centroid와의 neighborhoods points이다. 여기서 $K$는 KNN으로 고정된 값을 사용하였다. 

##### PointNet Layer

Input tensor $N' \times K \times (d+C)$를 PointNet에 넣어서 Output tensor $N' \times (d+C')$을 만들어낸다.
$N'$에 속해있는 각각의 centroid는 $K$개의 neighborhoods points와 각각 PointNet을 통과하여 결과를 만들어낸다.



#### Robust Feature Learning under Non-Uniform Sampling Density

![Pasted image 20240603130842](https://github.com/neporez/Pointcloud-Representation-Learning-Survey/assets/88701811/6e67dfe6-c361-42de-bbdf-9e746b197bd3)


일반적인 Pointcloud는 서로 다른 구역에서의 density가 non-uniform하고, 이러한 특성은 pointcloud의 feature learning에 있어서 좋지 않은 영향을 준다.

그렇기에 큰 Scale을 가지는 Pattern을 포착해야만 한다. 이전에 언급한 각각의 abstraction level layer는 단일 Scale의 feature를 추출했다. PointNet++에서는 각각의 abstraction level은 Multiple Scales의 feature를 통합하여 사용한다. Multiple Scales를 적용하는 방법에는 두가지 방법이 있다.

##### Multi-scale grouping(MSG)

Figure 3 (a)를 보면 서로 다른 Scale의 Region을 PointNet에 통과시켜 나온 Feature를 Concatenation하는 방식을 사용한다. 

##### Multi-resolution grouping(MRG)

MSG는 모든 point의 Multi Scale Region Feature를 PointNet에 통과시키기에 컴퓨터 자원을 많이 사용한다. 하지만, 이 특성은 유지하면서 많은 컴퓨터 자원을 사용하지 않는 방향의 접근법이 있다. Figure 3 (b)를 보면 상위 영역의 Feature와 상위 영역에 속해있는 Point들을 centroid로 하여 PointNet에 통과시켜 나온 하위 영역의 Feature를 Concatenation한다.

만약 밀도가 낮은 부근의 경우에는 하위 영역은 포인트가 부족하여 희박한 정보를 담고 있을 수 있기 때문에 하위 영역의 가중치를 높게 해야 한다.

#### Point Feature Propagation for Set Segmentation

set abstraction layer에서는 원본 데이터에서 일부가 샘플링되는 방식으로 진행된다. 하지만 Segmentation에서는 원본 Pointcloud의 모든 포인트에 Feature를 얻어내야만 한다. 한가지 해결책은 Pointcloud의 모든 포인트를 샘플링하여 set abstraction layer의 입력으로 넣어주는 것이다. 하지만 이는 많은 컴퓨터 자원을 소모한다.

다른 방법은 subsampled point에서 원본 point로의 원복이다.

Figure 2를 보면 Segmentation은 distance based interpolation과 각각의 set abstraction layer에서의 skip connection으로 이루어져 있다.

interpolation은 k nearest neighbors에 의해 다음과 같이 정해진다.

#### $f^{(j)}(x) = \frac{\sum^{k}_{i=1}w_i(x)f_i^{(j)}}{\sum^{k}_{i=1}w_i(x)} \quad where \quad w_i(x) = \frac{1}{d(x,x_i)^p}\, ,j = 1,...,C$

Interpolation이 끝나게 되면 각 포인트는 $1 \times 1$의 convolution과 비슷한 unit PointNet을 통과하게 되고 이때 skip connection으로 넘어오는 feature와 concatenation하게 된다.

---
## PointNet++의 특징

1.  Multi Scale Region의 Local feature를 사용하여 기존 PointNet의 문제인 Local Structure의 Feature를 포착하였다.

2.  Multi Scale Region의 Local feature의 계산을 용이하게 하기 위해 Multi-Resolution Grouping을 제안하였다.

3.  계층적인 구조로 Large Scale Region의 Feature부터 Small Scale Region의 Feature를 모두 포착하였고, 이를 Skip Connection을 통해 Feature의 손실을 막았다.

4.  Interpolation 작업을 통해 Sampling 된 Point들을 원복하여 Segmentation이 가능하도록 하였다.
---
## PointNet++의 장점

1.  Farthest Point Sampling을 통해 연산량을 줄이고, Point dropout을 통해 다양한 분포의 Pointcloud를 학습하도록 하여 다양한 상황에서 강건한 성능을 보여준다.

2.  다양한 Scale의 Local Region을 포착하기 때문에 기존 PointNet이 힘들어하던 작은 물체에 대한 정확도가 높아졌을 것이다. (작성자의 생각)
---
## PointNet++의 단점

1.  Point-based의 연산을 하기 때문에 기본적으로 추론 속도가 느리다.

2.  Local Region을 결정하는 과정에서 KNN을 사용하기 때문에 이는 많은 컴퓨터 자원을 사용한다.

3.  자율주행의 LiDAR와 같이 Sparse한 데이터에서 학습한 결과를 보여주지 않았고, 제시한 데이터셋은 모두 상대적으로 Dense하다.





















