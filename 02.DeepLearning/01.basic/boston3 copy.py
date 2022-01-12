import numpy as np
import sys
import warnings
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# argument 정리
if len(sys.argv) <= 1:          # argument 미입력
    print(sys.argv[0], len(sys.argv))
    print("사용법: python boston3.py test_dataset_index(0~50)")
    sys.exit()
try:
    index = int(sys.argv[1])
except:
    print("정수를 입력하세요.")
    print("사용법: python boston3.py test_dataset_index(0~50)")
    sys.exit()

if index < 0 or index > 50:
    print("0과 50사이의 정수를 입력하세요.")
    print("사용법: python boston3.py test_dataset_index(0~50)")
    sys.exit()     

# 상수값 설정, 변수 초기화
seed = 2022
model_filename = 'boston.h5'
warnings.filterwarnings('ignore')
np.random.seed(2022)
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, test_size = 0.1, random_state=seed
)
# 메인 모델 불러오기
model = load_model(model_filename)

# 메인

test = X_test[index].reshape(1,-1)
pred_value = model.predict(test)
print(f'실제값 : {y_test[index]}, 예측값:{pred_value[0,0]:.2f}')