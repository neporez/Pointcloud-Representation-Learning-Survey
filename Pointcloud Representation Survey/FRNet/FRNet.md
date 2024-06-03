
1. pointcloud+Image Projection 방식으로 post-processing을 없앴기에 속도가 빠르다.

2.  각 포인트마다의 Local Feature와 Frustum region마다의 Global Feature의 결합하여 모델의 정확성을 높였다.

3.  모든 Backbone network의 layer의 output을 활용하여 최종적인 결과를 만들어낸다.

4.  Auxiliary head를 통해 pseudo frustum label을 만들어내어 Loss function에 영향을 주었다.

Google Docs : https://docs.google.com/document/d/1Zi-fCmU9DA7s1KlyjwjDfJvA0043isabnwDPuwymSX4/edit?usp=sharing