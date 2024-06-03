## PointNet의 특징

1.  포인트 클라우드를 Image, Voxel로 변환하지 않고 모든 포인트를 활용하였다.

2.  각 포인트마다 Local Feature를 계산하고 Symmetric Function을 통해 순열 불변성을 지키면서 Global Feature를 계산한다.

3.  Point Segmentation을 위해 각 포인트의 Local Feature와 Global Feature를 Concatenating을 진행하여 각 포인트에 Local Geometry와 Global Semantics의 의미를 담았다.

4.  Affine transformation matrix를 orthogonal matrix로 추론하는 mini-network를 설계하여 feature extract 네트워크의 전처리 파트에 연결하였다.

Google Docs : https://docs.google.com/document/d/11vlczYdNxBp6taDS0KR6Vm0U-ix0voTALmywl7zVtpc/edit?usp=sharing
