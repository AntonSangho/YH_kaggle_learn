# Exercise 3: 최대 풀링 — 모범 답안

이 실습에서는 특성 추출의 마지막 단계인 **최대 풀링**을 학습합니다.

---

## Q1. 최대 풀링 적용

2×2 풀링 윈도우로 최대 풀링을 적용합니다.

```python
image_condense = tf.nn.pool(
    input=image_detect,
    window_shape=(2, 2),
    pooling_type='MAX',
    strides=(2, 2),
    padding='SAME',
)
```

> **설명**:
> | 파라미터 | 값 | 의미 |
> |---------|-----|------|
> | `window_shape` | `(2, 2)` | 2×2 풀링 윈도우 |
> | `pooling_type` | `'MAX'` | 윈도우 내 최댓값 선택 |
> | `strides` | `(2, 2)` | 2칸씩 이동 (이미지 크기 반으로) |
> | `padding` | `'SAME'` | 경계 처리로 크기 유지 |

---

## Q2. 평행이동 불변성 (Translation Invariance)

작은 크기의 평행이동에 대한 최대 풀링의 영향:

**관찰:**
- 최대 풀링을 여러 번 적용하면
- 작은 크기의 평행이동에도 불구하고
- 최종 결과가 비슷하게 유지됨

**이것이 중요한 이유:**
- 모델이 정확한 위치보다 **특성의 존재**에 집중
- 위치 변화에 강건한 분류 가능
- 객체가 약간 움직여도 인식 가능

```
원본 원 → 작은 변위 적용 → MaxPool 4회 → 비슷한 결과
(평행이동 불변성 확보)
```

---

## Q3. 풀링된 특성 해석

VGG16 기반의 전역 평균 풀링 결과:

**512개의 풀링된 특성값:**
- 각 값은 해당 특성 맵의 평균 픽셀값
- 차이가 충분히 큼 → 자동차와 트럭 구별 가능
- 완전히 파괴적이지만 효과적

**해석:**
```python
# 원본 특성 맵: 512개, 각각 크기 4×4 = 16 픽셀
# 전역 평균 풀링 후: 512개의 단일 값
# 파라미터 감소: 25배 (5×5 → 1 값)
```

---

## 풀링의 종류

| 풀링 | 수식 | 용도 | 특징 |
|------|------|------|------|
| **Max Pool** | `max(window)` | 기반(base) | 특성 강조, 번역 불변성 |
| **Avg Pool** | `mean(window)` | 기반(base) | 부드러운 특성 |
| **Global Max** | `max(feature_map)` | 헤드(head) | 특성의 최대값만 |
| **Global Avg** | `mean(feature_map)` | 헤드(head) | 특성의 평균값만 |

---

## GlobalAvgPool2D의 장점

```python
model = keras.Sequential([
    pretrained_base,              # VGG16 기반 (512개 특성 맵)
    layers.GlobalAvgPool2D(),    # 512개로 압축
    layers.Dense(1, activation='sigmoid'),  # 분류
])
```

**장점:**
1. **파라미터 감소**: 4×4×512 = 8,192 → 512
2. **계산 효율성**: Dense 층의 입력이 매우 작음
3. **정규화 효과**: 과적합 감소
4. **성능**: 전체 Flatten + Dense보다 종종 더 좋음

**단점:**
- 공간 정보 손실 (위치 정보 제거)
- 세밀한 공간 패턴 감지 불가

---

## 실습 데이터셋 분석

**Car or Truck 이미지의 풀링 특성:**
- 자동차 특성과 트럭 특성이 구별됨
- 색상, 질감, 형태 정보가 512개 값에 인코딩됨
- 이 512개 값으로 충분히 분류 가능

---

## 풀링 파라미터 해석

```python
tf.nn.pool(
    input=feature_map,          # 입력 (배치, 높이, 너비, 채널)
    window_shape=(2, 2),        # 풀링 윈도우 크기
    pooling_type='MAX',         # 최댓값 또는 평균
    strides=(2, 2),             # 윈도우 이동 크기
    padding='SAME'              # 'SAME' 또는 'VALID'
)
```

**Padding 비교:**
- `'SAME'`: 입력과 같은 크기 유지 (경계 처리)
- `'VALID'`: 경계 제외 (더 작은 출력)
