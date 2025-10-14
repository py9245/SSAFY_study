import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("study_data.csv", encoding="utf-8-sig")

print(df.head())

np.random.seed(24)

x = df.iloc[:, :-1].to_numpy(copy=False)
y = df.iloc[:, -1].to_numpy(copy=False)
x = np.c_[np.ones(x.shape[0]), x]
beta_1 = np.linalg.pinv(x.T @ x) @ x.T @ y


test_data = [1, 8, 2, 8]
test_data_2 = [1, 1.5973902572668779,4,1.2831604303925488]

# 기대했던 값과 비슷 경사 하강법 적용 후 검증해보겠음


lr = 0.007
epochs = 25000
beta = np.zeros(x.shape[1])
history = []

for i in range(epochs):
    y_pred = x @ beta
    error = y_pred - y
    grad = (1/len(y)) * (x.T @ error)
    beta -= lr * grad

    if i % 1000 == 0:
        mse = np.mean(error ** 2)
        history.append((i, mse, *beta))
print("beta:", beta)

# 오차 기록 그래프
epochs_list = [h[0] for h in history]
mse_list = [h[1] for h in history]

plt.plot(epochs_list, mse_list, marker='o')
plt.title("MSE 감소 그래프 (Gradient Descent)")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.show()


# 이건 정규 방정식 세타
test_data = [1, 8, 2, 8]
test_data_2 = [1, 2.9308947463495336, 4, 3.1036509383708584]

print("정규 방정식 예측값:", beta_1 @ test_data)
print("정규 방정식 예측값_2:", beta_1 @ test_data_2)

# 경사 하강법 구한 세타
pred1 = np.dot(beta, test_data)
pred2 = np.dot(beta, test_data_2)

print("경사 하강법 예측값:", pred1)
print("경사 하강법 예측값_2:", pred2)


