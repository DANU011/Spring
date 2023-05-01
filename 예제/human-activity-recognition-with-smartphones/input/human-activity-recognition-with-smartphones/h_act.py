import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터 로드 및 전처리
data = pd.read_csv('train.csv')
X = data.drop(['Activity', 'subject'], axis=1)
y = pd.get_dummies(data['Activity'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 2. XGBoost 모델 훈련
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# 3. 모델 평가 및 예측
test_data = pd.read_csv('test.csv')
X_test = test_data.drop(['Activity', 'subject'], axis=1)
y_test = pd.get_dummies(test_data['Activity'])
X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
