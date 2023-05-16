import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np

# 데이터 로드 및 전처리
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data.iloc[:, :-2].values
X_test = test_data.iloc[:, :-2].values
y_train = train_data.iloc[:, -1].values
y_test = test_data.iloc[:, -1].values

# 클래스 인코딩
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# XGBoost 모델 훈련
model = xgb.XGBClassifier()
model.fit(X_train, y_train)



# 예측
y_pred = model.predict(X_test)


# 결과 출력
y_pred = model.predict(X_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)
print(y_pred_decoded)

for i in range(len(y_pred)):
    print(f"Test sample {i+1}: Predicted={label_encoder.inverse_transform([y_pred[i]])[0]}, Actual={y_test[i]}")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# classification report
class_report = classification_report(y_test, y_pred)
print(class_report)
