import numpy as np
import time
from sklearn.datasets import make_blobs

# 하이퍼파라미터 설정
N = 1000
D = 8
num_classes = 3
hidden_dim = 16

# 고정된 centers 배열을 정의하여 학습 및 검증 데이터에 동일하게 적용합니다.
centers = np.array([
    np.full(D, -5.0),
    np.full(D, 0.0),
    np.full(D, 5.0)
])

# 학습 데이터 생성
X, y = make_blobs(n_samples=N, n_features=D, centers=centers, cluster_std=1.5, random_state=42)
y = y.reshape(-1, 1)

# 원-핫 인코딩 함수
def one_hot(y, num_classes):
    onehot = np.zeros((y.shape[0], num_classes))
    onehot[np.arange(y.shape[0]), y.flatten()] = 1
    return onehot

y_onehot = one_hot(y, num_classes)

# 소프트맥스, ReLU 및 미분 함수
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# 파라미터 초기화
np.random.seed(42)
W1 = np.random.randn(D, hidden_dim) * np.sqrt(2.0 / D)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim)
b2 = np.zeros((1, num_classes))

# Adam 하이퍼파라미터 설정
lr = 0.05
epochs = 1600
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Adam 모멘텀 변수 초기화
mW1 = np.zeros_like(W1)
vb1 = np.zeros_like(b1)
mW2 = np.zeros_like(W2)
vb2 = np.zeros_like(b2)
vW1 = np.zeros_like(W1)
vb1_v = np.zeros_like(b1)
vW2 = np.zeros_like(W2)
vb2_v = np.zeros_like(b2)

loss_history = []

start = time.perf_counter()
for epoch in range(1, epochs+1):
    # 순전파: 은닉층
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    # 순전파: 출력층
    Z2 = A1.dot(W2) + b2
    y_pred = softmax(Z2)
    
    # 손실 함수: 범주형 교차 엔트로피
    loss = -np.mean(np.sum(y_onehot * np.log(y_pred + 1e-8), axis=1))
    loss_history.append(loss)
    
    # 역전파: 출력층
    dZ2 = (y_pred - y_onehot)
    dW2 = A1.T.dot(dZ2) / N
    db2 = np.sum(dZ2, axis=0, keepdims=True) / N
    
    # 역전파: 은닉층
    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = X.T.dot(dZ1) / N
    db1 = np.sum(dZ1, axis=0, keepdims=True) / N
    
    # Adam 업데이트 (시간 단계 t = epoch)
    t = epoch
    mW1 = beta1 * mW1 + (1 - beta1) * dW1
    mW2 = beta1 * mW2 + (1 - beta1) * dW2
    vb1 = beta1 * vb1 + (1 - beta1) * db1
    vb2 = beta1 * vb2 + (1 - beta1) * db2
    vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
    vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
    vb1_v = beta2 * vb1_v + (1 - beta2) * (db1 ** 2)
    vb2_v = beta2 * vb2_v + (1 - beta2) * (db2 ** 2)
    
    mW1_corr = mW1 / (1 - beta1 ** t)
    mW2_corr = mW2 / (1 - beta1 ** t)
    vb1_corr = vb1 / (1 - beta1 ** t)
    vb2_corr = vb2 / (1 - beta1 ** t)
    vW1_corr = vW1 / (1 - beta2 ** t)
    vW2_corr = vW2 / (1 - beta2 ** t)
    vb1_v_corr = vb1_v / (1 - beta2 ** t)
    vb2_v_corr = vb2_v / (1 - beta2 ** t)
    
    W1 -= lr * mW1_corr / (np.sqrt(vW1_corr) + epsilon)
    b1 -= lr * vb1_corr / (np.sqrt(vb1_v_corr) + epsilon)
    W2 -= lr * mW2_corr / (np.sqrt(vW2_corr) + epsilon)
    b2 -= lr * vb2_corr / (np.sqrt(vb2_v_corr) + epsilon)
    
    if epoch % 200 == 0 or epoch == epochs:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

end = time.perf_counter()
print("Adam 학습 코드 실행 시간: {:.4f} 초".format(end - start))
print("학습 완료")

# 검증 데이터 생성 및 평가 (학습 데이터와 동일한 centers를 사용)
N_val = int(N * 0.5)
X_val, y_val = make_blobs(n_samples=N_val, n_features=D, centers=centers, cluster_std=1.5, random_state=42)
y_val = y_val.reshape(-1, 1)
y_val_onehot = one_hot(y_val, num_classes)

Z1_val = X_val.dot(W1) + b1
A1_val = relu(Z1_val)
Z2_val = A1_val.dot(W2) + b2
y_pred_val = softmax(Z2_val)
val_loss = -np.mean(np.sum(y_val_onehot * np.log(y_pred_val + 1e-8), axis=1))
print("Validation Loss: {:.4f}".format(val_loss))
