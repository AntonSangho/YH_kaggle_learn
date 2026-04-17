# Exercise 1: 컨볼루션 분류기 — 모범 답안

이 실습에서는 사전학습된(pretrained) 모델을 기반으로 전이학습(transfer learning)을 수행하여 이미지 분류 모델을 만듭니다.

---

## Q1. 사전학습 기반 설정

사전학습된 InceptionV1 모델의 가중치를 다시 학습할지 결정하는 문제입니다.

```python
pretrained_base.trainable = False
```

> **설명**: 전이학습을 할 때 사전학습된 기반(base)을 다시 훈련하면 안 됩니다. 그 이유는:
> - 새로운 헤드의 무작위 가중치가 큰 그래디언트 업데이트를 생성
> - 이것이 기반 레이어로 역전파되어 사전학습 정보를 파괴함
> - `trainable = False`로 설정하면 기반의 특성 추출 능력을 보존

---

## Q2. 조밀 층 헤드 추가

사전학습된 기반 위에 분류용 헤드를 구성합니다.

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])
```

> **설명**: 
> - `Flatten()`: 3D 특성 맵을 1D 벡터로 변환
> - `Dense(6, activation='relu')`: 은닉층 (6개 뉴런, ReLU 활성화)
> - `Dense(1, activation='sigmoid')`: 출력층 (이진 분류이므로 1개 뉴런, 시그모이드)

---

## Q3. 모델 컴파일

이진 분류 문제에 적합한 손실 함수와 메트릭을 선택합니다.

```python
optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
```

> **설명**:
> | 구성 요소 | 선택 | 이유 |
> |---------|------|------|
> | Loss | `binary_crossentropy` | 이진 분류 문제 (Car=0, Truck=1) |
> | Metric | `binary_accuracy` | 이진 분류의 정확도 측정 |
> | Optimizer | `Adam` | 일반적으로 안정적인 성능 제공 |

---

## Q4. 손실과 정확도 곡선 분석

**관찰 포인트:**
- **VGG16 대비 InceptionV1의 장점**:
  - 과적합(overfitting)이 덜함
  - 검증 손실이 더 안정적
  - 일찍부터 좋은 검증 정확도 달성

- **이것이 의미하는 것**:
  - InceptionV1 아키텍처가 이 데이터셋에 더 적합
  - 더 효율적인 특성 추출
  - 정규화 효과가 더 좋음

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **전이학습** | 한 작업에서 학습한 지식을 다른 작업에 적용 |
| **사전학습 기반** | ImageNet 등으로 미리 학습된 모델 부분 |
| **특성 추출** | 기반이 수행하는 역할 (이미지의 패턴 학습) |
| **분류 헤드** | 추출된 특성으로부터 클래스 예측하는 부분 |
| **Binary Crossentropy** | 이진 분류의 표준 손실 함수 |

---

## 학습 곡선 해석

```
훈련 정확도: 90.68%
검증 정확도: 85.98%
```

- 훈련 정확도 > 검증 정확도: 약간의 과적합 (정상 범위)
- 30 에포크에서 수렴: 충분한 훈련
- 검증 손실이 안정적: 모델이 일반화 가능
