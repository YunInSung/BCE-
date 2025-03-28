import numpy as np
import time
from sklearn.datasets import make_blobs

# 하이퍼파라미터 설정
N = 50000           # 샘플 수
D = 8               # 입력 차원
num_classes = 4     # 클래스 수
hidden_dim1 = 16    # 첫 번째 은닉층 크기
hidden_dim2 = 8     # 두 번째 은닉층 크기

# 고정된 centers 배열 (클러스터 중심)
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

# 활성화 함수 및 미분 함수
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 파라미터 초기화 (He 초기화)
np.random.seed(42)
W1 = np.random.randn(D, hidden_dim1) * np.sqrt(2.0 / D)
b1 = np.zeros((1, hidden_dim1))
W2 = np.random.randn(hidden_dim1, hidden_dim2) * np.sqrt(2.0 / hidden_dim1)
b2 = np.zeros((1, hidden_dim2))
W3 = np.random.randn(hidden_dim2, num_classes) * np.sqrt(2.0 / hidden_dim2)
b3 = np.zeros((1, num_classes))

# Adam 하이퍼파라미터 설정
lr = 0.25
epochs = 800
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Adam 모멘텀 변수 초기화
mW1, vb1 = np.zeros_like(W1), np.zeros_like(b1)
mW2, vb2 = np.zeros_like(W2), np.zeros_like(b2)
mW3, vb3 = np.zeros_like(W3), np.zeros_like(b3)
vW1, vW2, vW3 = np.zeros_like(W1), np.zeros_like(W2), np.zeros_like(W3)
vb1_v, vb2_v, vb3_v = np.zeros_like(b1), np.zeros_like(b2), np.zeros_like(b3)

loss_history = []

start = time.perf_counter()
for epoch in range(1, epochs+1):
    # 순전파: 첫 번째 은닉층
    Z1 = X.dot(W1) + b1         # (N, hidden_dim1)
    A1 = relu(Z1)
    
    # 순전파: 두 번째 은닉층
    Z2 = A1.dot(W2) + b2        # (N, hidden_dim2)
    A2 = relu(Z2)
    
    # 순전파: 출력층
    Z3 = A2.dot(W3) + b3        # (N, num_classes)
    y_pred = softmax(Z3)
    
    # 손실 함수: 범주형 교차 엔트로피
    loss = -np.mean(np.sum(y_onehot * np.log(y_pred + 1e-8), axis=1))
    loss_history.append(loss)
    
    # 역전파: 출력층
    dZ3 = (y_pred - y_onehot)              # (N, num_classes)
    dW3 = A2.T.dot(dZ3) / N                # (hidden_dim2, num_classes)
    db3 = np.sum(dZ3, axis=0, keepdims=True) / N
    
    # 역전파: 두 번째 은닉층
    dA2 = dZ3.dot(W3.T)                    # (N, hidden_dim2)
    dZ2 = dA2 * relu_deriv(Z2)             # (N, hidden_dim2)
    dW2 = A1.T.dot(dZ2) / N                # (hidden_dim1, hidden_dim2)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / N
    
    # 역전파: 첫 번째 은닉층
    dA1 = dZ2.dot(W2.T)                    # (N, hidden_dim1)
    dZ1 = dA1 * relu_deriv(Z1)             # (N, hidden_dim1)
    dW1 = X.T.dot(dZ1) / N                 # (D, hidden_dim1)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / N
    
    # Adam 업데이트 (시간 단계 t = epoch)
    t = epoch
    # 1차 모멘트 업데이트
    mW1 = beta1 * mW1 + (1 - beta1) * dW1
    mW2 = beta1 * mW2 + (1 - beta1) * dW2
    mW3 = beta1 * mW3 + (1 - beta1) * dW3
    vb1 = beta1 * vb1 + (1 - beta1) * db1
    vb2 = beta1 * vb2 + (1 - beta1) * db2
    vb3 = beta1 * vb3 + (1 - beta1) * db3
    
    # 2차 모멘트 업데이트
    vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
    vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
    vW3 = beta2 * vW3 + (1 - beta2) * (dW3 ** 2)
    vb1_v = beta2 * vb1_v + (1 - beta2) * (db1 ** 2)
    vb2_v = beta2 * vb2_v + (1 - beta2) * (db2 ** 2)
    vb3_v = beta2 * vb3_v + (1 - beta2) * (db3 ** 2)
    
    # 편향 보정 (Bias correction)
    mW1_corr = mW1 / (1 - beta1 ** t)
    mW2_corr = mW2 / (1 - beta1 ** t)
    mW3_corr = mW3 / (1 - beta1 ** t)
    vb1_corr = vb1 / (1 - beta1 ** t)
    vb2_corr = vb2 / (1 - beta1 ** t)
    vb3_corr = vb3 / (1 - beta1 ** t)
    
    vW1_corr = vW1 / (1 - beta2 ** t)
    vW2_corr = vW2 / (1 - beta2 ** t)
    vW3_corr = vW3 / (1 - beta2 ** t)
    vb1_v_corr = vb1_v / (1 - beta2 ** t)
    vb2_v_corr = vb2_v / (1 - beta2 ** t)
    vb3_v_corr = vb3_v / (1 - beta2 ** t)
    
    # 파라미터 업데이트
    W1 -= lr * mW1_corr / (np.sqrt(vW1_corr) + epsilon)
    b1 -= lr * vb1_corr / (np.sqrt(vb1_v_corr) + epsilon)
    W2 -= lr * mW2_corr / (np.sqrt(vW2_corr) + epsilon)
    b2 -= lr * vb2_corr / (np.sqrt(vb2_v_corr) + epsilon)
    W3 -= lr * mW3_corr / (np.sqrt(vW3_corr) + epsilon)
    b3 -= lr * vb3_corr / (np.sqrt(vb3_v_corr) + epsilon)
    
    if epoch == 1 or epoch == 50:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
    if epoch % 200 == 0 or epoch == epochs:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

end = time.perf_counter()
print("Adam 학습 코드 실행 시간: {:.4f} 초".format(end - start))
print("학습 완료")

# 검증 데이터 생성 및 평가 (학습 데이터와 동일한 centers 사용)
N_val = int(N * 0.5)
X_val, y_val = make_blobs(n_samples=N_val, n_features=D, centers=centers, cluster_std=1.5, random_state=42)
y_val = y_val.reshape(-1, 1)
y_val_onehot = one_hot(y_val, num_classes)

Z1_val = X_val.dot(W1) + b1
A1_val = relu(Z1_val)
Z2_val = A1_val.dot(W2) + b2
A2_val = relu(Z2_val)
Z3_val = A2_val.dot(W3) + b3
y_pred_val = softmax(Z3_val)
val_loss = -np.mean(np.sum(y_val_onehot * np.log(y_pred_val + 1e-8), axis=1))
print("Validation Loss: {:.6f}".format(val_loss))
