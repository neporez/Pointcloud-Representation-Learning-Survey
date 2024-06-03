
1. hand craft representation이 아닌 machine-learned feature representation 방식으로 end-to-end Learning 방식이다.

2. point level의 feature와 voxel level의 feature의 결합을 통해 voxel feature를 생성하여 Conv2D를 사용한다.

3. GPU의 효율적인 사용을 위해 모든 Voxel을 사용하지 않고,  dense voxel structure를 구성하여 빠른 추론 속도를 지향했다.

4. RPN를 통해 target의 bounding box를 구하는 1 stage dection 기법이다.

Google Docs : https://docs.google.com/document/d/1O8tHG-9VToXzjKT-j8KC624x9JRn-isZFwiwu8QhjmA/edit