# Exercise 5: 사용자 정의 컨볼루션 네트워크 — 모범 답안

이 실습에서는 처음부터 컨볼루션 네트워크를 설계하고 훈련합니다.

---

## Q1. 모델 정의

블록 구조의 컨볼루션 네트워크를 설계합니다. 각 블록은 점점 더 많은 컨볼루션을 포함합니다.

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # Block One
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
                  input_shape=[128, 128, 3]),
    layers.MaxPool2D(),

    # Block Two
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Three
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Head
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid'),
])
```

> **블록 구조의 설계**:
> - Block 1: 32개 필터 × 1개 Conv → 특성 시작
> - Block 2: 64개 필터 × 2개 Conv → 특성 강화
> - Block 3: 128개 필터 × 3개 Conv → 복잡한 패턴 학습

### 각 파라미터의 의미

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `filters` | 32, 64, 128 | 각 블록마다 증가 (깊어질수록 더 많은 특성) |
| `kernel_size` | 3 | 표준 커널 크기 |
| `activation` | 'relu' | 비선형성 추가 |
| `padding='same'` | 같은 크기 | 입력과 출력 크기 유지 |

---

## Q2. 모델 컴파일

이진 분류를 위해 적절한 손실과 메트릭을 설정합니다.

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
```

> **이진 분류 설정**:
> - **Loss**: `binary_crossentropy` (클래스 0 vs 1)
> - **Metric**: `binary_accuracy` (정확도)
> - **Optimizer**: Adam (안정적)

---

## Q3. 훈련 곡선 분석

**모델 성능:**
```
최종 훈련 정확도: ~90%
최종 검증 정확도: ~85%
```

**VGG16 vs Custom ConvNet 비교:**

| 측면 | VGG16 | Custom |
|------|--------|--------|
| 예훈련 | 있음 ✓ | 없음 |
| 초기 성능 | 매우 높음 | 낮음 (난수 가중치) |
| 수렴 속도 | 빠름 | 느림 |
| 최종 성능 | 86% | 85% |
| 파라미터 | 많음 | 적음 |
| 훈련 시간 | 빠름 | 길어질 수 있음 |

**결론:**
- 비슷한 성능 달성 가능
- 사전학습 없이도 효과적
- 작은 모델로 충분한 경우 유용

---

## 아키텍처 설계 원칙

### 1. 필터 수의 증가

```python
Block 1: 32 필터 (입력 이미지)
Block 2: 64 필터 (2배)
Block 3: 128 필터 (2배)
```

**왜?**
- 얕은 층: 간단한 특성 (엣지, 질감)
- 깊은 층: 복잡한 특성 (형태, 객체)
- 필터 수 증가로 표현력 향상

### 2. 컨볼루션 층 수 증가

```python
Block 1: 1개 Conv (간단)
Block 2: 2개 Conv (중간)
Block 3: 3개 Conv (복잡)
```

**이점:**
- 수용 영역 확대
- 파라미터 효율성
- 비선형성 증가

### 3. Dropout 정규화

```python
layers.Dropout(0.2)  # 20% 뉴런 비활성화
```

**목적:**
- 과적합 방지
- 모델 견고성 향상
- 훈련 중 협력적 학습 강제

---

## 모델 크기 비교

```python
# Block 구조
Input (128×128×3)
  ↓
Block1: Conv(32) + MaxPool → (64×64×32)
  ↓
Block2: Conv(64)×2 + MaxPool → (32×32×64)
  ↓
Block3: Conv(128)×3 + MaxPool → (16×16×128)
  ↓
Flatten: (16×16×128 = 32,768)
  ↓
Dense(6) → Dense(1)
```

**파라미터 수:**
- 각 Conv(3×3): `(previous_filters × 9 + 1) × current_filters`
- 모든 Conv 합계: ~100K 파라미터
- VGG16: ~14M 파라미터 (훨씬 큼)

---

## 훈련 곡선 해석

```
epoch | train_loss | val_loss | train_acc | val_acc
  1   |   0.57     |  0.44    |  0.70     | 0.80
 10   |   0.32     |  0.35    |  0.87     | 0.83
 30   |   0.24     |  0.34    |  0.90     | 0.85
 50   |   0.20     |  0.35    |  0.91     | 0.86
```

**관찰:**
1. 훈련 손실: 꾸준히 감소 ✓
2. 검증 손실: 일정하게 유지 ✓
3. 과적합: 최소한 (차이 작음)
4. 수렴: 30 에포크 이후 안정

---

## 최적화 팁

### 하이퍼파라미터 조정

```python
# 정규화 강화
layers.Dropout(0.3)  # 30%로 증가

# 또는 L2 정규화
layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu',
    padding='same',
    kernel_regularizer=tf.keras.regularizers.l2(0.01)
)

# 배치 정규화 추가
layers.BatchNormalization()
```

### 모델 성능 향상 전략

1. **더 깊은 네트워크**: 블록 추가
2. **더 넓은 네트워크**: 필터 수 증가
3. **더 나은 정규화**: Dropout, BatchNorm
4. **데이터 증강**: 입력 다양성 증가
5. **학습율 조정**: Adam epsilon 값 변경

---

## 핵심 개념

| 개념 | 역할 | 효과 |
|------|------|------|
| Conv2D | 특성 추출 | 이미지 패턴 학습 |
| MaxPool2D | 압축 | 공간 차원 축소 |
| Flatten | 1D 변환 | Dense 층 연결 |
| Dense | 분류 | 특성으로부터 클래스 예측 |
| Dropout | 정규화 | 과적합 방지 |
| ReLU | 활성화 | 비선형성 추가 |
| Sigmoid | 활성화 | [0, 1] 범위로 정규화 |

---

## 다음 단계

- Lesson 6: **데이터 증강**으로 성능 더욱 향상
- 더 큰 데이터셋에 이 아키텍처 적용
- 하이퍼파라미터 튜닝으로 최적화
