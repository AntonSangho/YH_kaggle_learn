# Exercise 4: 슬라이딩 윈도우 — 모범 답안

이 실습에서는 컨볼루션과 풀링이 이미지를 어떻게 스캔하는지 이해합니다.

---

## Q1. 수용 영역의 성장

3개의 연속된 3×3 컨볼루션 층의 수용 영역(receptive field):

**계산:**
- 1번째 Conv (3×3) → 수용 영역: **3×3**
- 2번째 Conv (3×3) → 수용 영역: **5×5** 
- 3번째 Conv (3×3) → 수용 영역: **7×7**

```
계산식: RF_n = RF_{n-1} + (kernel_size - 1)
3번째: 5 + (3 - 1) = 7
```

> **중요한 통찰**:
> - 3개의 3×3 커널: 27개 파라미터
> - 1개의 7×7 커널: 49개 파라미터
> - 같은 수용 영역, 더 적은 파라미터 ✓

---

## 슬라이딩 윈도우 파라미터

```python
visiontools.show_extraction(
    image, kernel,
    conv_stride=1,          # 컨볼루션 이동 크기
    conv_padding='valid',   # 패딩: 'valid' 또는 'same'
    pool_size=2,            # 풀링 윈도우 크기
    pool_stride=2,          # 풀링 이동 크기
    pool_padding='same',    # 풀링 패딩
    subplot_shape=(1, 4),
    figsize=(14, 6),
)
```

### 각 파라미터의 영향

| 파라미터 | 값 | 효과 |
|---------|-----|------|
| `conv_stride` | 1 | 꼼꼼한 특성 추출 (큰 출력) |
| `conv_stride` | 2+ | 빠른 처리 (작은 출력, 정보 손실) |
| `conv_padding='same'` | 입력과 같은 크기 | 경계 정보 보존 |
| `conv_padding='valid'` | 경계 제외 | 경계 정보 손실, 더 작은 출력 |
| `pool_stride` | window_size | 중첩 없음 (효율적) |
| `pool_stride` | 1 | 중첩 있음 (정보 풍부) |

---

## 1D 컨볼루션 (시계열)

시계열 데이터에도 컨볼루션을 적용할 수 있습니다.

```python
# 시계열 커널 예제
detrend = tf.constant([-1, 1], dtype=tf.float32)
average = tf.constant([0.2, 0.2, 0.2, 0.2, 0.2], dtype=tf.float32)
spencer = tf.constant([-3, -6, -5, 3, 21, ...], dtype=tf.float32) / 320

# 1D 컨볼루션 적용
ts_filter = tf.nn.conv1d(
    input=ts_data,
    filters=kern,
    stride=1,
    padding='VALID',
)
```

### 커널의 의미

**Detrend ([-1, 1]):**
- 인접한 두 시점의 차이 계산
- **시계열의 변화** 감지
- 추세 제거

**Average ([0.2, 0.2, 0.2, 0.2, 0.2]):**
- 5개 시점의 평균
- 높은 주파수 성분 제거
- **매끄러운 곡선** 생성

**Spencer (복잡한 커널):**
- 더 정교한 평활
- 특정 주파수 성분 추출

---

## 수용 영역의 중요성

**깊은 네트워크의 이점:**
```
얕은 네트워크:
Conv → Conv → Conv (3개 층)
각 단계마다 수용 영역 증가

깊은 네트워크:
Conv → Conv → ... → Conv (많은 층)
더 큰 수용 영역으로 큰 패턴 감지
```

**파라미터 효율성:**
- 3×3 커널 3개: 27 params (7×7 수용 영역)
- 7×7 커널 1개: 49 params (7×7 수용 역)
- **깊은 구조가 더 효율적!**

---

## Computer Vision의 다양한 데이터

### 2D 이미지
```python
# RGB 이미지
image.shape = (height, width, 3)  # 채널 3개

# 그레이스케일
image.shape = (height, width, 1)  # 채널 1개

# Conv2D: 2D 커널 적용
tf.nn.conv2d(image, kernel_2d)
```

### 1D 시계열
```python
# 시계열 데이터
ts.shape = (time_steps,)  # 시간 축만

# Conv1D: 1D 커널 적용
tf.nn.conv1d(ts, kernel_1d)
```

### 3D 비디오
```python
# 비디오: 시공간 데이터
video.shape = (height, width, frames)

# Conv3D: 3D 커널 적용 (공간 + 시간)
tf.nn.conv3d(video, kernel_3d)
```

---

## Google Trends 데이터 분석

**Machine Learning 검색 인기도 (2015-2020):**
- 2016-2017: 급격한 증가
- 2018-2019: 안정화
- 2019-2020: 재상승

**컨볼루션의 용도:**
- 추세 탐지 (detrend)
- 노이즈 제거 (average, spencer)
- 미래 예측 모델 학습

---

## 핵심 정리

| 차원 | 데이터 | 커널 | 함수 | 용도 |
|------|--------|------|------|------|
| 1D | 시계열 | 1×1 배열 | `conv1d()` | 추세, 패턴 |
| 2D | 이미지 | 3×3 배열 | `conv2d()` | 특성 추출 |
| 3D | 비디오 | 3×3×3 배열 | `conv3d()` | 시공간 특성 |

컨볼루션 네트워크는 모든 차원의 데이터에 적용 가능하며, 본질은 동일합니다.
