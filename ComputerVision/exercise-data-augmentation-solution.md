# Exercise 6: 데이터 증강 — 모범 답안

이 실습에서는 데이터 증강으로 모델 성능을 향상시킵니다.

---

## Q1. EuroSAT 데이터셋의 적절한 변환

위성 이미지 분류를 위한 적절한 데이터 증강:

**추천 변환:**
```python
augment = keras.Sequential([
    preprocessing.RandomFlip(mode='horizontal'),  # ✓
    preprocessing.RandomFlip(mode='vertical'),    # ✓
    preprocessing.RandomRotation(factor=0.2),     # ✓
    preprocessing.RandomContrast(factor=0.2),     # ✓
])
```

**선택 이유:**
| 변환 | 적절성 | 이유 |
|------|--------|------|
| 수평 뒤집기 | ✓ 적절 | 위성 이미지는 방향 중립적 |
| 수직 뒤집기 | ✓ 적절 | 위성 이미지는 방향 중립적 |
| 회전 | ✓ 적절 | 지표의 방향은 임의적 |
| 명도 조정 | ✓ 적절 | 계절/시간에 따른 변화 |
| 너비/높이 스트레치 | ✗ 부적절 | 왜곡되어 지표 형태 변경 |

---

## Q2. TensorFlow Flowers 데이터셋의 적절한 변환

꽃 사진 분류를 위한 적절한 데이터 증강:

**추천 변환:**
```python
augment = keras.Sequential([
    preprocessing.RandomFlip(mode='horizontal'),  # ✓
    preprocessing.RandomRotation(factor=0.2),     # ✓
    preprocessing.RandomContrast(factor=0.2),     # ✓
    preprocessing.RandomTranslation(
        height_factor=0.1, 
        width_factor=0.1
    ),                                             # ✓
])
```

**선택 이유:**
| 변환 | 적절성 | 이유 |
|------|--------|------|
| 수평 뒤집기 | ✓ 적절 | 꽃은 좌우 대칭 |
| 수직 뒤집기 | ✗ 부적절 | 꽃은 위/아래 비대칭 |
| 회전 | ✓ 적절 | 꽃의 방향은 임의적 |
| 명도 조정 | ✓ 적절 | 조명 조건 변화 |
| 평행이동 | ✓ 적절 | 프레임 내 위치 변화 |
| 너비 스트레치 | ⚠️ 주의 | 약간의 perspective 변화 |

---

## Q3. 전처리 층 추가

모델 시작에 데이터 증강을 통합합니다.

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

model = keras.Sequential([
    layers.InputLayer(input_shape=[128, 128, 3]),
    
    # Data Augmentation
    preprocessing.RandomContrast(factor=0.10),
    preprocessing.RandomFlip(mode='horizontal'),
    preprocessing.RandomRotation(factor=0.10),

    # Block One
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Two
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Three
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Head
    layers.BatchNormalization(renorm=True),
    layers.Flatten(),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])
```

### 주요 개선사항:

**1. 전처리 층의 위치:**
```python
Input
  ↓
[데이터 증강] ← 훈련 중에만 적용!
  ↓
[나머지 모델]
```

**2. BatchNormalization 추가:**
- `renorm=True`: 배치 정규화 재정규화
- 데이터 증강과 함께 사용하면 안정성 증가

**3. 더 큰 필터 수:**
- 이전: 32, 64 필터
- 지금: 64, 128, 256 필터
- 이유: 데이터 증강으로 과적합 위험 감소

---

## Q4. 훈련 곡선 분석

**데이터 증강의 효과:**

```
                | 증강 전 | 증강 후
시작 성능       | 70%    | 약간 낮음 (증강 노이즈)
최종 성능       | 85%    | 87-89%
과적합          | 있음   | 감소
수렴 안정성     | 약간   | 더 안정적
에포크당 시간   | 빠름   | 느림
```

### 관찰 포인트:

**과적합 감소:**
```
증강 전:
- epoch 1: train_acc=70%, val_acc=65% (차이 5%)
- epoch 50: train_acc=95%, val_acc=85% (차이 10%) ← 과적합

증강 후:
- epoch 1: train_acc=65%, val_acc=62% (차이 3%)
- epoch 50: train_acc=90%, val_acc=88% (차이 2%) ← 덜 과적합
```

**성능 개선:**
- 훈련 세트의 다양성 증가
- 모델의 일반화 능력 향상
- 검증 정확도 2-4% 향상

---

## 데이터 증강의 원리

### 효과적인 증강의 조건

```python
# ✓ 좋은 예
augment = keras.Sequential([
    preprocessing.RandomFlip(mode='horizontal'),
    preprocessing.RandomRotation(factor=0.1),
    preprocessing.RandomContrast(factor=0.1),
])

# ✗ 나쁜 예
augment = keras.Sequential([
    preprocessing.RandomRotation(factor=0.5),  # 너무 큼
    preprocessing.RandomTranslation(height_factor=0.5),  # 너무 극단적
])
```

### 파라미터 설정 가이드

| 파라미터 | 범위 | 추천 | 주의사항 |
|---------|------|------|----------|
| `factor` | 0.0~1.0 | 0.1~0.2 | 너무 크면 레이블 변경 |
| `height_factor` | 0.0~1.0 | 0.1 | 이미지 심각한 변형 주의 |
| `width_factor` | 0.0~1.0 | 0.1 | 객체 인식성 유지 필요 |

---

## 데이터 증강 기법 비교

| 기법 | 원본 | 증강 후 | 용도 |
|------|------|--------|------|
| **Flip** | 🚗 | 🚗 ↔️ | 방향 불변성 |
| **Rotation** | 🚗 | 🚗⟳ | 회전 불변성 |
| **Contrast** | 🚗 | 🚗(밝음) | 조명 변화 |
| **Translation** | 🚗 | 🚗(이동) | 위치 변화 |
| **Width/Height** | 🚗 | 🚗(늘어남) | 관점 변화 |

---

## 실무 팁

### 데이터 증강을 사용해야 할 때

✓ **사용 권장:**
- 훈련 데이터가 부족
- 과적합 발생
- 실제 데이터의 변화가 큼
- 계산 자원이 충분

✗ **사용 비권장:**
- 데이터가 매우 많음
- 데이터 변환이 레이블 변경
- 실시간 성능이 중요

### 증강 전략

```python
# 보수적 증강 (안전)
augment_safe = keras.Sequential([
    preprocessing.RandomFlip(mode='horizontal'),
    preprocessing.RandomContrast(factor=0.1),
])

# 공격적 증강 (더 많은 다양성)
augment_aggressive = keras.Sequential([
    preprocessing.RandomFlip(mode='horizontal'),
    preprocessing.RandomFlip(mode='vertical'),
    preprocessing.RandomRotation(factor=0.2),
    preprocessing.RandomContrast(factor=0.2),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
])
```

---

## 모델 성능 요약

**3개 모델 비교 (Car or Truck):**

| 모델 | 구조 | 기반 | 정확도 | 시간 |
|------|------|------|--------|------|
| Lesson 1 | InceptionV1 + Dense | 사전학습 | 86% | 빠름 |
| Lesson 5 | Custom ConvNet | 없음 | 85% | 중간 |
| Lesson 6 | Custom + 증강 | 없음 | 87-89% | 느림 |

**결론:**
- 데이터 증강으로 custom 모델이 사전학습 모델 성능 추월
- 계산 비용이 가장 크지만 효과적
- 실무에서 매우 유용한 기법

---

## 핵심 정리

| 항목 | 설명 |
|------|------|
| **목적** | 훈련 데이터 다양성 증가 |
| **위치** | 모델 시작 (첫 번째 층) |
| **적용 시점** | 훈련 중에만 (testing=False일 때만) |
| **파라미터** | 데이터셋에 따라 조정 필요 |
| **주의** | 증강이 레이블을 변경하면 안 됨 |
| **효과** | 과적합 감소, 성능 향상 |

데이터 증강은 현대 딥러닝의 필수 기법입니다!
