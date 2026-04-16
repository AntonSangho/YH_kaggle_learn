# 이진 분류

딥러닝을 또 다른 일반적인 작업에 적용해 봅시다.

## 소개
지금까지 이 강좌에서 우리는 신경망이 회귀 문제를 어떻게 해결할 수 있는지 배웠습니다. 이제 신경망을 또 다른 일반적인 기계 학습 문제인 분류에 적용해 보겠습니다. 지금까지 배운 내용의 대부분은 여전히 유효합니다. 주된 차이점은 사용하는 손실 함수와 최종 레이어가 생성해야 하는 출력의 유형에 있습니다.

## 이진 분류
두 개의 클래스 중 하나로 분류하는 것은 흔한 기계 학습 문제입니다. 고객이 구매할 가능성이 있는지 없는지, 신용카드 거래가 사기인지 아닌지, 심우주 신호가 새로운 행성의 증거인지, 또는 의료 검사 결과가 질병의 증거인지 예측하고 싶을 수 있습니다. 이 모든 것이 이진 분류 문제입니다.

원시 데이터에서 클래스는 “예”와 ‘아니오’, 또는 “개”와 “고양이”와 같은 문자열로 표현될 수 있습니다. 이 데이터를 사용하기 전에 클래스 레이블을 할당해야 합니다. 한 클래스는 0, 다른 클래스는 1로 지정합니다. 숫자 레이블을 할당하면 데이터를 신경망이 사용할 수 있는 형태로 변환하게 됩니다.

## 정확도와 교차 엔트로피
정확도는 분류 문제의 성공 여부를 측정하는 데 사용되는 여러 지표 중 하나입니다. 정확도는 전체 예측 중 정답의 비율을 의미합니다: 정확도 = 정답 수 / 총 예측 수. 항상 정확하게 예측하는 모델의 정확도 점수는 1.0이 됩니다. 다른 조건이 동일하다면, 데이터셋 내 클래스가 대략 비슷한 빈도로 나타날 때 정확도는 합리적인 지표로 사용할 수 있습니다.

정확도(및 대부분의 다른 분류 지표)의 문제점은 이를 손실 함수로 사용할 수 없다는 것입니다. SGD는 부드럽게 변화하는 손실 함수가 필요하지만, 정확도는 개수의 비율이기 때문에 “단순히 뚝 떨어지는” 방식으로 변화합니다. 따라서 손실 함수 역할을 할 대안을 선택해야 합니다. 이 대안이 바로 교차 엔트로피 함수입니다.

이제, 손실 함수가 훈련 중 네트워크의 목표를 정의한다는 점을 상기해 봅시다. 회귀 분석의 경우, 우리의 목표는 기대 결과와 예측 결과 사이의 거리를 최소화하는 것이었습니다. 우리는 이 거리를 측정하기 위해 MAE를 선택했습니다.

분류의 경우, 우리가 원하는 것은 확률 간의 거리이며, 이것이 바로 교차 엔트로피가 제공하는 것입니다. 교차 엔트로피는 한 확률 분포에서 다른 확률 분포까지의 거리를 측정하는 일종의 척도입니다.

[16.png]

* 교차 엔트로피는 부정확한 확률 예측에 대해 페널티를 부과합니다.

핵심은 네트워크가 올바른 클래스를 1.0의 확률로 예측하도록 하는 것입니다. 예측된 확률이 1.0에서 멀어질수록 교차 엔트로피 손실값은 커집니다.

교차 엔트로피를 사용하는 기술적인 이유는 다소 미묘하지만, 이 섹션에서 기억해야 할 핵심은 바로 이것입니다. 분류 손실 함수로는 교차 엔트로피를 사용하십시오. 그러면 정확도와 같이 중요하게 여길 수 있는 다른 지표들도 함께 개선되는 경향이 있습니다.


## 시그모이드 함수를 이용한 확률 생성
크로스 엔트로피와 정확도 함수는 모두 확률, 즉 0에서 1 사이의 숫자를 입력으로 필요로 합니다. 밀집층에서 생성된 실수 값 출력을 확률로 변환하기 위해, 우리는 새로운 종류의 활성화 함수인 시그모이드 활성화 함수를 적용합니다.

[17.png]

최종 클래스 예측 결과를 얻기 위해 임계값 확률을 정의합니다. 일반적으로 이 값은 0.5로 설정되며, 이렇게 하면 반올림 시 올바른 클래스가 산출됩니다. 즉, 0.5 미만이면 레이블 0인 클래스로, 0.5 이상이면 레이블 1인 클래스로 분류됩니다. Keras는 정확도 지표 계산 시 기본적으로 0.5 임계값을 사용합니다.


# 예시 - 이진 분류
자, 이제 직접 해봅시다!

이온스피어 데이터셋에는 지구 대기의 이온스피어 층을 대상으로 한 레이더 신호에서 추출한 특징들이 포함되어 있습니다. 이 작업의 목표는 신호가 특정 물체의 존재를 나타내는지, 아니면 단순히 공허한 공간인지 판단하는 것입니다.

```python 
import pandas as pd
from IPython.display import display

ion = pd.read_csv('../input/dl-course-data/ion.csv', index_col=0)
display(ion.head())

df = ion.copy()
df['Class'] = df['Class'].map({'good': 0, 'bad': 1})

df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)

df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)
df_train.dropna(axis=1, inplace=True) # drop the empty feature in column 2
df_valid.dropna(axis=1, inplace=True)

X_train = df_train.drop('Class', axis=1)
X_valid = df_valid.drop('Class', axis=1)
y_train = df_train['Class']
y_valid = df_valid['Class']
```

회귀 작업에서 했던 것과 똑같이 모델을 정의하되, 한 가지 예외가 있습니다. 모델이 클래스 확률을 산출할 수 있도록 최종 레이어에 ‘시그모이드’ 활성화 함수를 포함시킵니다.

```python 
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[33]),
    layers.Dense(4, activation='relu'),    
    layers.Dense(1, activation='sigmoid'),
])
```

모델의 compile 메서드를 사용하여 크로스 엔트로피 손실과 정확도 지표를 모델에 추가하세요. 2-클래스 문제의 경우 반드시 ‘binary’ 버전을 사용해야 합니다. (클래스가 더 많은 문제는 약간 다르게 처리됩니다.) Adam 최적화기는 분류 문제에서도 매우 효과적이므로, 계속 이 최적화기를 사용할 것입니다.

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
```

이 특정 문제의 모델은 훈련을 완료하는 데 꽤 많은 에포크가 소요될 수 있으므로, 편의상 조기 종료 콜백을 포함하겠습니다.


```python 
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)
```
평소와 같이 학습 곡선을 살펴보고, 검증 세트에서 얻은 손실값과 정확도 중 최적의 값을 확인해 보겠습니다. (조기 종료(early stopping)를 적용하면 가중치가 이 값들을 얻었을 때의 상태로 복원된다는 점을 기억해 주세요.)

```python
history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))
```

```
Best Validation Loss: 0.3534
Best Validation Accuracy: 0.8857
```
[18.png]
[19.png]

‘Hotel Cancellations’ 데이터셋을 사용하여 신경망을 통해 호텔 예약 취소 여부를 예측해 보세요.


