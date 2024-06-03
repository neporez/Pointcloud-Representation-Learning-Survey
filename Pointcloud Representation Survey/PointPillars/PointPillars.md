
1. range view, bird’s eye view를 사용하는 다른 image projection 방식과 다르게 pillars의 단위로 pseudo-image를 만들어내었다.

2. 모든 포인트 클라우드를 사용하는 것이 아닌, 특정 개수로 한정된 포인트들을 사용하여서 dense한 tensor를 만들어 GPU의 연산효율을 높였다.

3. Detection Head로 SSD를 채택하여서 1-stage detection이 가능토록 하였다.

4. Inference 속도가 62Hz에 육박하고 작은 버전의 network는 105Hz까지 도달한다.

Google Docs : https://docs.google.com/document/d/1QG1fdGetYKvm7T3Az2CCbVSQyuuNMxB72weiEj5qqSg/edit