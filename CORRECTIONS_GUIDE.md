# 한국어 번역 수정 가이드

> 각 레슨의 구체적인 수정 방법을 라인별로 제시합니다.

---

## Lesson 4: 슬라이딩 윈도우 🔴 우선순위 1

**파일**: `ComputerVision/4_TheSlidingWindow_korean.md`  
**심각도**: 🔴 매우 높음  
**예상 수정 시간**: 35분  
**문제 수**: 6개

### 문제 1: circle() 함수 들여쓰기 (Lines 8-22)

**현재 상태 (❌)**:
```python
def circle(size, val=None, r_shrink=0):
    circle = np.zeros([size[0]+1, size[1]+1])
    
rr, cc = draw.circle_perimeter(
        size[0]//2, size[1]//2,
        radius=size[0]//2 - r_shrink,
        shape=[size[0]+1, size[1]+1],
    )
    if val is None:
        
circle[rr, cc] = np.random.uniform(size=circle.shape)[rr, cc]
    else:
        circle[rr, cc] = val
    circle = transform.resize(circle, size, order=0)
    return circle
```

**수정된 상태 (✅)**:
```python
def circle(size, val=None, r_shrink=0):
    circle = np.zeros([size[0]+1, size[1]+1])
    
    rr, cc = draw.circle_perimeter(
        size[0]//2, size[1]//2,
        radius=size[0]//2 - r_shrink,
        shape=[size[0]+1, size[1]+1],
    )
    if val is None:
        circle[rr, cc] = np.random.uniform(size=circle.shape)[rr, cc]
    else:
        circle[rr, cc] = val
    circle = transform.resize(circle, size, order=0)
    return circle
```

**변경 사항**:
- Line 11: `rr, cc = draw.circle_perimeter(` 앞에 4칸 들여쓰기 추가
- Line 18: `circle[rr, cc] = ...` 앞에 8칸 들여쓰기 추가 (if 블록 내)

---

### 문제 2: show_kernel() 함수 들여쓰기 (Lines 24-48)

**현재 상태 (❌)**:
```python
def show_kernel(kernel, label=True, digits=None, text_size=28):
    
# 커널 형식 지정
    kernel = np.array(kernel)
    if digits is not None:
        kernel = kernel.round(digits)

    # 커널 플롯
    cmap = plt.get_cmap('Blues_r')
    plt.imshow(kernel, cmap=cmap)
    
rows, cols = kernel.shape
    thresh = (kernel.max()+kernel.min())/2
    # 선택적으로 값 레이블 추가
    if label:
        for i, j in product(range(rows), range(cols)):
            val = kernel[i, j]
            color = cmap(0) if val > thresh else cmap(255)
            
plt.text(j, i, val, 
                     color=color, size=text_size,
                     horizontalalignment='center', verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
```

**수정된 상태 (✅)**:
```python
def show_kernel(kernel, label=True, digits=None, text_size=28):
    # 커널 형식 지정
    kernel = np.array(kernel)
    if digits is not None:
        kernel = kernel.round(digits)

    # 커널 플롯
    cmap = plt.get_cmap('Blues_r')
    plt.imshow(kernel, cmap=cmap)
    
    rows, cols = kernel.shape
    thresh = (kernel.max()+kernel.min())/2
    # 선택적으로 값 레이블 추가
    if label:
        for i, j in product(range(rows), range(cols)):
            val = kernel[i, j]
            color = cmap(0) if val > thresh else cmap(255)
            
            plt.text(j, i, val, 
                     color=color, size=text_size,
                     horizontalalignment='center', verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
```

**변경 사항**:
- Line 26: `# 커널 형식 지정` 앞에 4칸 들여쓰기 추가
- Line 35: `rows, cols = kernel.shape` 앞에 4칸 들여쓰기 추가
- Line 43: `plt.text(j, i, val,` 앞에 12칸 들여쓰기 추가 (for 루프 내)

---

### 문제 3: show_extraction() 함수 - 파라미터 레이아웃 (Lines 49-61)

**현재 상태 (❌)**:
```python
def show_extraction(image,
                    kernel,
                    conv_stride=1,
                    conv_padding='valid',
                    activation='relu',
                    pool_size=2,
                    pool_stride=2,
                    pool_padding='same',
                    figsize=(10, 10),
                    subplot_shape=(2, 2),
                    
ops=['Input', 'Filter', 'Detect', 'Condense'],
                    gamma=1.0):
```

**수정된 상태 (✅)**:
```python
def show_extraction(image,
                    kernel,
                    conv_stride=1,
                    conv_padding='valid',
                    activation='relu',
                    pool_size=2,
                    pool_stride=2,
                    pool_padding='same',
                    figsize=(10, 10),
                    subplot_shape=(2, 2),
                    ops=['Input', 'Filter', 'Detect', 'Condense'],
                    gamma=1.0):
```

**변경 사항**:
- Line 60: `ops=['Input', 'Filter', 'Detect', 'Condense'],` 을 line 59 끝에 함께 배치하거나, 올바르게 들여쓰기
- 또는 다음과 같이 정렬 (권장):
```python
def show_extraction(image,
                    kernel,
                    conv_stride=1,
                    conv_padding='valid',
                    activation='relu',
                    pool_size=2,
                    pool_stride=2,
                    pool_padding='same',
                    figsize=(10, 10),
                    subplot_shape=(2, 2),
                    ops=['Input', 'Filter', 'Detect', 'Condense'],
                    gamma=1.0):
```

---

### 문제 4: tf.keras. Sequential 공백 오류 (Line 63)

**현재 상태 (❌)**:
```python
    model = tf.keras. Sequential([
```

**수정된 상태 (✅)**:
```python
    model = tf.keras.Sequential([
```

**변경 사항**: `tf.keras.` 와 `Sequential` 사이의 공백 제거

---

### 문제 5: tf.keras.layers. Activation 공백 오류 (Line 72)

**현재 상태 (❌)**:
```python
                    tf.keras.layers. Activation(activation),
```

**수정된 상태 (✅)**:
```python
                    tf.keras.layers.Activation(activation),
```

**변경 사항**: `tf.keras.layers.` 와 `Activation` 사이의 공백 제거

---

### 문제 6: 수식 포맷 손상 (Lines 249-255, 297-302)

**문제 A - Line 249-255**

**현재 상태 (❌)**:
```
VGG 아키텍처는 상당히 간단합니다. 스트라이드 1의 컨볼루션과 
2
×
2
2
×
2
 윈도우 및 스트라이드 2의 맥시멈 풀링을 사용합니다.
```

**수정된 상태 (✅)**:
```
VGG 아키텍처는 상당히 간단합니다. 스트라이드 1의 컨볼루션과 2×2 윈도우 및 스트라이드 2의 맥시멈 풀링을 사용합니다.
```

또는 긴 문장이면:
```
VGG 아키텍처는 상당히 간단합니다. 스트라이드 1의 컨볼루션과 2×2 윈도우 및 스트라이드 2의 맥시멈 풀링을 사용합니다.
```

**문제 B - Line 297-302**

**현재 상태 (❌)**:
```
ResNet50 모델은 첫 번째 레이어에서 스트라이드 2의 
7
×
7
7
×
7
 커널을 사용합니다.
```

**수정된 상태 (✅)**:
```
ResNet50 모델은 첫 번째 레이어에서 스트라이드 2의 7×7 커널을 사용합니다.
```

**변경 사항**: 수식을 한 줄로 통합, 중복된 "7×7"를 하나로 변경

---

### 추가 개선사항

**Line 237 - 용어 수정 (선택사항)**

**현재 상태**:
```
# 하단 소벨
```

**권장 수정**:
```
# 소벨 필터
```

---

## Lesson 2: 컨볼루션과 ReLU ⚠️ 우선순위 2

**파일**: `ComputerVision/2_ConvolutionandReLU_kroean.md`  
**심각도**: ⚠️ 높음  
**예상 수정 시간**: 25분  
**문제 수**: 3개

### 문제 1: show_kernel() 함수 들여쓰기 (Lines 7-30)

**현재 상태 (❌)**:
```python
def show_kernel(kernel, label=True, digits=None, text_size=28):
    # 커널 형식 지정
    kernel = np.array(kernel)
    if digits is not None:
        kernel = kernel.round(digits)
    
# 커널 플롯
    cmap = plt.get_cmap('Blues_r')
    plt.imshow(kernel, cmap=cmap)
    rows, cols = kernel.shape
    thresh = (kernel.max()+kernel.min())/2
    # 선택적으로 값 레이블 추가
    
if label:
        for i, j in product(range(rows), range(cols)):
            val = kernel[i, j]
            color = cmap(0) if val > thresh else cmap(255)
            plt.text(j, i, val, 
                     color=color, size=text_size,
                     horizontalalignment='center', verticalalignment='center')
    
plt.xticks([])
    plt.yticks([])
```

**수정된 상태 (✅)**:
```python
def show_kernel(kernel, label=True, digits=None, text_size=28):
    # 커널 형식 지정
    kernel = np.array(kernel)
    if digits is not None:
        kernel = kernel.round(digits)
    
    # 커널 플롯
    cmap = plt.get_cmap('Blues_r')
    plt.imshow(kernel, cmap=cmap)
    rows, cols = kernel.shape
    thresh = (kernel.max()+kernel.min())/2
    # 선택적으로 값 레이블 추가
    
    if label:
        for i, j in product(range(rows), range(cols)):
            val = kernel[i, j]
            color = cmap(0) if val > thresh else cmap(255)
            plt.text(j, i, val, 
                     color=color, size=text_size,
                     horizontalalignment='center', verticalalignment='center')
    
    plt.xticks([])
    plt.yticks([])
```

**주요 변경**:
- Line 13: `# 커널 플롯` 앞에 4칸 들여쓰기 추가
- Line 20: `if label:` 앞에 4칸 들여쓰기 추가
- Line 28: `plt.xticks([])` 앞에 4칸 들여쓰기 추가

---

### 문제 2: tf.nn.conv2d() 들여쓰기 (Lines 187-194)

**현재 상태 (❌)**:
```python
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    # 이 두 가지에 대해서는 4강에서 다룰 예정입니다!
    
strides=1,
    padding='SAME',
)
```

**수정된 상태 (✅)**:
```python
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    # 이 두 가지에 대해서는 4강에서 다룰 예정입니다!
    strides=1,
    padding='SAME',
)
```

**변경 사항**:
- Line 192: `strides=1,` 앞에 4칸 들여쓰기 추가
- 라인 191의 빈 줄 제거 (선택사항)

---

### 문제 3: ReLU 용어 검토 (Lines 106, 110, 이후)

**현재 상태**:
```
리크티파이어 함수의 그래프는 다음과 같습니다:

리크티파이어가 연결된 뉴런을 리크티파이드 리니어 유닛(ReLU)이라고 합니다.
```

**권장 수정**:
```
ReLU(Rectified Linear Unit) 함수의 그래프는 다음과 같습니다:

이러한 함수가 연결된 뉴런을 ReLU 유닛(ReLU, Rectified Linear Unit)이라고 합니다.
```

**설명**:
- "리크티파이어"는 직역이지만, 한국어 기술 문서에서는 "ReLU"를 그대로 사용하는 것이 표준
- 영어 약자와 함께 표기하면 독자 이해도 높아짐

---

## Lesson 5: 사용자 정의 컨볼루션 신경망 ⚠️ 우선순위 2

**파일**: `ComputerVision/5_CustomConvnets_korean.md`  
**심각도**: ⚠️ 중간  
**예상 수정 시간**: 20분  
**문제 수**: 4개

### 문제 1: set_seed() 함수 들여쓰기 (Lines 44-50)

**현재 상태 (❌)**:
```python
def set_seed(seed=31415):
    
np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()
```

**수정된 상태 (✅)**:
```python
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()
```

**변경 사항**:
- Line 45의 빈 줄 제거
- Line 46: `np.random.seed(seed)` 앞에 4칸 들여쓰기 추가

---

### 문제 2: .map() 공백 오류 (Line 87)

**현재 상태 (❌)**:
```python
    .map (convert_to_float)
```

**수정된 상태 (✅)**:
```python
    .map(convert_to_float)
```

**변경 사항**: `map`과 `(` 사이의 공백 제거

---

### 문제 3: Conv2D input_shape 들여쓰기 (Lines 119-123)

**현재 상태 (❌)**:
```python
    layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same',
                  # 첫 번째 레이어의 입력 차원을 지정합니다
                  # [높이, 너비, 색상 채널(RGB)]
                  
input_shape=[128, 128, 3]),
```

**수정된 상태 (✅) - 방법 1 (권장)**:
```python
    layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same',
                  input_shape=[128, 128, 3]),
```

**수정된 상태 (✅) - 방법 2 (자세한 주석 유지)**:
```python
    layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same',
                  # 첫 번째 레이어의 입력 차원을 지정합니다
                  # [높이, 너비, 색상 채널(RGB)]
                  input_shape=[128, 128, 3]),
```

**변경 사항**:
- Line 123: `input_shape=[128, 128, 3]),` 앞에 18칸 들여쓰기 추가 (또는 방법 1처럼 간단히)

---

### 문제 4: 모델 요약 출력 포맷 (Lines 150-157)

**현재 상태 (❌)**:
```python
 max_pooling2d (MaxPooling2D (None, 64, 64, 32) 0 
 )
...
 max_pooling2d_1 (MaxPooling (None, 32, 32, 64) 0
  
2D)
```

**참고**: 이 부분은 `model.summary()` 출력의 문자열이므로, 
원본 코드가 올바르다면 실행 시 자동으로 생성되는 부분입니다.

만약 마크다운 파일에서 수작업으로 수정해야 한다면, 다음과 같이:

**수정된 상태 (✅)**:
```python
 max_pooling2d (MaxPooling2D) (None, 64, 64, 32) 0
 
 conv2d_1 (Conv2D) (None, 64, 64, 64) 18496
 
 max_pooling2d_1 (MaxPooling2D) (None, 32, 32, 64) 0
```

**또는** 원본 코드 블록이 올바르다면, 실제 실행 결과를 다시 캡처할 것 권장.

---

## Lesson 1: 컨볼루션 분류기 ✅ 우선순위 3 (낮음)

**파일**: `ComputerVision/1_TheConvolutionalClassifier_korean.md`  
**심각도**: ✅ 낮음  
**예상 수정 시간**: 2분  
**문제 수**: 1개

### 문제 1: set_seed() 공백 오류 (Line 83)

**현재 상태 (❌)**:
```python
    np.random.seed (seed)
```

**수정된 상태 (✅)**:
```python
    np.random.seed(seed)
```

**변경 사항**: `seed`와 `(` 사이의 공백 제거

---

## Lesson 6: 데이터 증강 ✅ 우선순위 3 (낮음)

**파일**: `ComputerVision/6_DataAugmentation_korean.md`  
**심각도**: ✅ 낮음  
**예상 수정 시간**: 3분  
**문제 수**: 2개

### 문제 1: set_seed() 함수 들여쓰기 (Lines 48-54)

**현재 상태 (❌)**:
```python
def set_seed(seed=31415):
    
np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()
```

**수정된 상태 (✅)**:
```python
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()
```

**변경 사항**:
- Line 49의 빈 줄 제거
- Line 50: `np.random.seed(seed)` 앞에 4칸 들여쓰기 추가

---

### 문제 2: .map() 공백 오류 (Line 91)

**현재 상태 (❌)**:
```python
    .map (convert_to_float)
```

**수정된 상태 (✅)**:
```python
    .map(convert_to_float)
```

**변경 사항**: `map`과 `(` 사이의 공백 제거

---

## Lesson 3: 최대 풀링 ✅ 유지 (수정 불필요)

**파일**: `ComputerVision/3_MaximumPooling_korean.md`  
**상태**: 🟢 완벽한 상태  
**조치**: 변경 없음

---

## 📋 수정 체크리스트

### Phase 1: 긴급 수정 (Lesson 4)
- [ ] Lesson 4 - circle() 함수 들여쓰기
- [ ] Lesson 4 - show_kernel() 함수 들여쓰기
- [ ] Lesson 4 - show_extraction() 파라미터 레이아웃
- [ ] Lesson 4 - tf.keras. Sequential 공백 제거
- [ ] Lesson 4 - tf.keras.layers. Activation 공백 제거
- [ ] Lesson 4 - 수식 포맷 통합

### Phase 2: 높은 우선순위 (Lesson 2, 5)
- [ ] Lesson 2 - show_kernel() 들여쓰기
- [ ] Lesson 2 - tf.nn.conv2d() 들여쓰기
- [ ] Lesson 2 - ReLU 용어 검토
- [ ] Lesson 5 - set_seed() 들여쓰기
- [ ] Lesson 5 - .map() 공백 제거
- [ ] Lesson 5 - Conv2D input_shape 들여쓰기
- [ ] Lesson 5 - 모델 요약 포맷 확인

### Phase 3: 낮은 우선순위 (Lesson 1, 6)
- [ ] Lesson 1 - np.random.seed() 공백 제거
- [ ] Lesson 6 - set_seed() 들여쓰기
- [ ] Lesson 6 - .map() 공백 제거

---

**예상 총 수정 시간**: 60-80분

