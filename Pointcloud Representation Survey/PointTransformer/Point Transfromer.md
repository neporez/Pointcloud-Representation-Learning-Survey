
# Point Transformer

Zhao, H., Jiang, L., Jia, J., Torr, P. H., & Koltun, V. (2021). Point transformer. In _Proceedings of the IEEE/CVF international conference on computer vision_ (pp. 16259-16268).

---

## Abstract

Self-attention network는 자연어 처리와 이미지 처리(Object Detection, Classification 등)에서 혁신적인 성과를 이뤄냈다. 이 성과에 영감을 받아, 본 논문에서는 3D pointcloud를 처리하는 self-attention network를 제안한다. 이 network는 semantic segmentation, object classification 등의 task에서 활용될 수 있다. S3DIS Dataset Area 5에서 mIoU 70.4%의 이전 연구의 모델보다 더 좋은 성과를 얻었다.

---

## Introduction

3D pointcloud가 자율주행, 로보틱스 등에서 자주 활용됨.

Pointcloud는 이미지와 다르게 곧바로 CNN과 같은 네트워크를 적용할 수 없음

이전에 Pointcloud를 다루고자하는 다양한 연구가 있었음.

그 중 Voxel-based의 단점을 보완하기 위해 sparse convolution이 사용됨

혹은, pooling operator와 같은 symmetric function을 사용하는 네트워크나, Graph CNN이 활용되었다고함

Transformer Network의 핵심인 Self-attention 기법은 set operator(Symmetric function)이기 때문에 Pointcloud processing에 특히 적합하다. (symmetric function에 대한 이야기는 PointNet에서 자세히 다룸)

본 네트워크에서는 local neighborhoods끼리 self-attention을 적용하였고, S3DIS에서 좋은 성능을 보였다.

---

## Point Transformer

우선, 일반적인 형태의 transformer에서 사용되는 self-attention을 소개하고, 본 네트워크에서 사용되는 point transformer layer에 대해서 소개한다.

### Background

Self-attention operator는 두 가지로 나눠진다.

#### Scalar attention

$\chi = \{x_i\}_i$ 일 때, Standard scalar dot-product attention layer는 다음과 같이 쓸 수 있다.

####  $y_i = \sum\limits_{x_j \in \chi} \rho(\varphi(x_i)^\top\psi(x_j)+\delta)\alpha(x_j),$

$y_i$는 output feature이고, $\varphi, \psi, \alpha$는 pointwise feature transformation(MLP)이다.(좀 더 자세히 말하자면 Transformer 개념에 등장하는 query, key, value network라고 생각하면 된다.)

$\delta$는 Position Encoding, $\rho$는 normalization function(softmax)이다.

$\rho(\varphi(x_i)^\top\psi(x_j)+\delta))$ 이 파트의 output을 attention weight라고 한다.

#### Vector attention

Vector attention에서는 attention weight  ${\large Vectors}$ 로 서로 형태가 다르다.

#### $y_i = \sum\limits_{x_j \in\chi} \rho(\gamma(\beta(\varphi(x_i),\psi(x_j))+ \delta))\odot\alpha(x_j),$

$\beta$는 relation function(Substraction)이고, $\gamma$는 mapping function(MLP)이다.

Scalr attention, Vector attention 둘 다 set operator이며, 이 operator는 전체 signal을 표현하거나, 일부 local path의 signal을 표현하기도 한다. (self attention이 문장 혹은 이미지의 전체 맥락을 판단한다는 의미)

### Point Transformer Layer

본 논문에서 사용되는 point transformer layer는 vector attention을 기반으로 만들어졌다. relation function은 substraction이고, positional encoding $\delta$는 attention vector $\gamma$와 transformed feature $\alpha$(value)에 각각 더해진다.

![Pasted image 20240625125334](https://github.com/neporez/Pointcloud-Representation-Learning-Survey/assets/88701811/71695f19-6124-4c9a-9b51-5cdffc9f4cd1)

#### $y_i = \sum\limits_{x_j\in\chi(i)}\rho(\gamma(\varphi(x_i)-\psi(x_j)+\delta))\odot(\alpha(x_j)+\delta)$

여기서 subset $\chi(i)\subseteq \chi$은 pointcloud $\chi$의 point인 $x_i$의 local neighborhood이다.(K nearest neighbors)

따라서 전체 pointcloud에 대해서 self-attention을 적용하는 것이 아닌, $x_i$의 local neightborhood끼리 local self-attention을 통해 $x_i$의 feature를 계산한다.


### Position Encoding

position encoding은 self-attention에서 operator에 local structure의 정보를 적용하는 중요한 역할을 맡는다. text sequence, Image에서 일반적인 position encoding은 hand-craft(e.g., sine and cosine function)을 통해 만들어진다. 3D pointcloud processing에서는 pointcloud 스스로의 coordinate가 position encoding의 후보가 되는 것이 자연스럽다. 따라서 position encoding은 다음과 같다.

#### $\delta=\theta(p_i-p_j)$.

여기서 $p_i, p_j$는 각각 i,j번째 point의 coordinate이고 $\theta$는 MLP이다. 주목할만한 것은 본 논문에서 attention generation branch와 feature transformation branch 둘 모두에서 position encoding이 중요하다는 사실이다. 따라서 두 branch 모두에서 추가되었고, end-to-end network에서 subnetwork에서 parameter $\theta$를 학습한다.

### Point Transformer Block

![Pasted image 20240625135352](https://github.com/neporez/Pointcloud-Representation-Learning-Survey/assets/88701811/a1ca0c23-6bcd-4546-a373-44275831817c)

본 논문에서 point transformer block은 self-attention layer가 포함된 point transformer 앞 뒤로 linear layer를 포함하고 있고, residual connection 또한 포함하고 있다.

### Network Architecture


![Pasted image 20240626103146](https://github.com/neporez/Pointcloud-Representation-Learning-Survey/assets/88701811/f70504e0-aa11-460d-b348-9caf25b75eea)

![Pasted image 20240626104549](https://github.com/neporez/Pointcloud-Representation-Learning-Survey/assets/88701811/64a9d39f-baad-41bc-97d6-b6d07962c315)


Point Transformer는 point transformer block을 제외하고도 두가지 모듈이 추가적으로 사용되었다.


#### Transition down

transition down 모듈의 키 포인트는 point set의 cardinality(카디널리티)를 줄이는 것이다.(여기서 말하는 cardinality는 어떤 데이터 셋에서 유니크한 값이 많을수록 높다고 지칭하는 말이다. 즉, 카디널리티가 줄어든다는 것은 <유니크한 값의 개수가 줄어듬 == 포인트를 샘플링한다> 라는 의미로 작성자가 해석함)

예를 들어서, 첫번째 블록에서 두번째 블록으로 입력값이 통과할 때, N개의 포인트에서 N/4개로 포인트를 샘플링하여 갯수가 줄어든다. (N,C) -> (N/4, C)

여기서 KNN을 통해 각 포인트와 가까운 K개의 포인트를 선별한다. (N/4,C) -> (N/4,K,C)

이를 MLP에 통과시키고 (N/4,K,C) -> (N/4,K,C)

Local max pooling을 통해 축소시킨다. (N/4,K,C) -> (N/4,C) 
-> 이때 많이 사용하는 함수가 torch_scatter의 scatter_max 인듯하다. 혹은 torch 2.3 버전에 존재하는 torch.Tensor.scatter_reduce 함수를 사용할 수도 있음


#### Transition up

Point Transformer는 Image segmentation의 대표적인 모델인 U-Net 구조를 적용시켰다. 따라서, Point Transformer는 Transition down block과 point transformer block으로 이루어진 encoder가 있고, 그와 symmetric 구조를 띄는 transition up block, point transformer block으로 이루어진 decoder가 존재한다. 

Upsampling시에는 trilinear interpolation을 통한 high-resolution pointcloud와 skip connection을 통해 encoder의 각 레이어에서 넘어온 high-resolution pointcloud를 summation한다.


#### Output head

segmentation을 위해서는 decoder의 마지막 블록에서 feature vector가 부여된 각각의 point들을 MLP에 통과시켜 class의 개수 차원으로 feature를 줄여야한다. Classification은 U-Net 구조가 아닌 Encoder의 마지막 블록에서 Global AvgPooling을 사용하고 이를 MLP에 통과시켜 최종 클래스의 개수만큼 차원을 줄인다.

---

## Point Transformer의 특징

1. 각각의 포인트에 Neighborhood의 정보를 압축시키는 vector attention operator를 활용하였다.

2. 연산을 줄이기 위해 Farthest point sampling을 활용하여 point의 cardinality를 줄였다.

3. U-Net 구조를 적용하여 Encoder-Decoder 구조를 띄고 있다.

4. 당시 기준 S3DIS Dataset에서 SOTA를 달성하였다.

5. 이후 나오는 Transformer 계열의 논문들에서 이 논문의 단점을 해결하는 종류의 논문이 많이 등장한다.

---

## Point Transformer의 장점

1. Pointcloud 자체를 입력으로 받고, 후처리 또한 필요 없는 end-to-end Model이다.

2. 상대적으로 Dense한 Pointcloud, Sparse한 Pointcloud에서 모두 적용 가능하다.

---

## Point Transformer의 단점


1. KNN을 활용하여 Sampling이후 Neighborhood를 찾기 때문에 Inference 속도가 느려서 실제 자율주행에서는 사용하기 힘들다.

2. Point의 개수가 많아질수록 KNN으로 인해 연산량이 크게 늘어난다,

3. encoder와 decoder 모두 KNN이 포함되기 때문에 모델의 확장 역시 쉽지 않다.
