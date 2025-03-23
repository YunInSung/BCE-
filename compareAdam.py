import numpy as np
import matplotlib.pyplot as plt
import time

N = 50000          # 데이터 샘플 수
D = 10            # 입력 차원
hidden_dim = 25
size = N
epsilon = 1e-9
learn = 0.03
iterator = 100
######################################################################################################
######################################################################################################
######################################################################################################
def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def sigmoid_dx(x) :
    return np.exp(-x) / (1 + np.exp(-x))**2

def sigmoid_d2x(x) :
    return (np.exp(-2 * x) - np.exp(-x)) / (1 + np.exp(-x))**3

# def relu(x):
#     return np.maximum(0, x)

# def relu_deriv(x):
#     return (x > 0).astype(np.float32)

# def relu_deriv2(x):
#     return 0

def relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def relu_deriv(x, alpha=0.01):
    return np.where(x >= 0, 1, alpha)

def relu_deriv2(x, alpha=0.01):
    return np.zeros_like(x)

######################################################################################################
######################################################################################################
######################################################################################################

def g_func(x) :
    return sigmoid_dx(x) / (sigmoid(x) + epsilon)
def g_func_dx(x) :
    return (sigmoid_d2x(x) * sigmoid(x) - sigmoid_dx(x) ** 2) / (sigmoid(x) ** 2 + epsilon)
def h_func(x) :
    return sigmoid_dx(x) / (1 - sigmoid(x) + epsilon)
def h_func_dx(x) :
    return (sigmoid_d2x(x) * (1 -  sigmoid(x)) + sigmoid_dx(x) ** 2) / ((1 - sigmoid(x)) ** 2 + epsilon)

def zeta_dx(z, y) : 
    return (y * g_func(z) - (1 - y) * h_func(z))
def zeta_d2x(z, y) :
    return (y * g_func_dx(z) - (1 - y) * h_func_dx(z))

#####################################
#############  W2 b2  ###############
#####################################


def grad_W2_func_r(h, row, d2Z2, A1i):
    new_h = d2Z2[row] * h / N
    result = new_h[:,np.newaxis,:] * A1i[np.newaxis,:,:]
    matrix = np.sum(result, axis=2).T
    return matrix

def H_matirx2_r(h, row, d2Z2, A1i, grad_W2_r):
    m = h.shape[0]
    matrix = np.hstack([grad_W2_r, np.sum(d2Z2[row] * A1i, axis = 1).reshape(m+1,1) / N])
    return matrix

def L_matrix2_r(h, P, row, dW2, db2, d2Z2, A1i, grad_W2_r):
    m = h.shape[0]
    v0 = P[row]
    W2_r =  dW2[row] * learn
    matrix = (v0@grad_W2_r) - W2_r
    grad_b_r = np.sum(d2Z2[row] * A1i, axis = 1).reshape(m+1,1) / N
    b_r = db2[row] * learn
    tmp = v0.reshape(1, m + 1).dot(grad_b_r) - b_r
    matrix = np.append(matrix, tmp).reshape(1, m + 1)
    return matrix

#####################################
#############  W1 b1  ###############
#####################################


def grad_W1_func_r(theta, X, Xi):
    tmp = theta * X / N
    result = tmp[:,np.newaxis,:] * Xi[np.newaxis,:,:]
    matrix = np.sum(result, axis=2).T
    return matrix

def H_matirx1_r(theta, X, Xi, grad_W1_r):
    n = X.shape[0]
    matrix = np.hstack([grad_W1_r, np.sum(theta * Xi, axis = 1).reshape(n + 1, 1) / N])
    return matrix

def L_matrix1_r(X, P, theta, dW1, db1, row, Xi, grad_W1_r):
    n = X.shape[0]
    v0 = P[row]
    W1_r =  dW1[row] * learn
    matrix = (v0.dot(grad_W1_r)) - W1_r
    grad_b_r = np.sum(theta * Xi, axis = 1).reshape(n + 1, 1) / N
    b_r = db1[row] * learn
    tmp = v0.reshape(1, n + 1).dot(grad_b_r) - b_r
    matrix = np.append(matrix, tmp).reshape(1, n + 1)
    return matrix

def P_matrix(X, Y, W1, b1, W2, b2):
    matrix2 = 0
    matrix1 = 0
    p = Y.shape[0]
    m = W1.shape[0]
    n = X.shape[0]
    cpW1 = W1.copy()
    cpb1 = b1.copy()
    cpW2 = W2.copy()
    cpb2 = b2.copy()

    y = Y
    Z1 = W1@X + b1
    A1 = relu(Z1)
    Z2 = W2@A1 + b2
    y_pred = sigmoid(Z2)
    dZ2 = y_pred - y            # (N, 1)
    d2Z2 = y_pred * (1 - y_pred)
    dW2 = dZ2.dot(A1.T) / N     # (hidden_dim, 1)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / N  # (1, 1)
    # 은닉층: dL/dA1 = dZ2 dot W2^T, dL/dZ1 = dL/dA1 * ReLU'(Z1)
    dA1 = W2.T@dZ2         # (N, hidden_dim)
    dZ1 = dA1 * relu_deriv(Z1)  # (N, hidden_dim)
    dW1 = dZ1.dot(X.T) / N      # (D, hidden_dim)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / N  # (1, hidden_dim)

    i = np.ones((1, X.shape[1]))
    Xi = np.vstack((X, i))
    P = np.hstack([W1, b1])
    df1 = relu_deriv(Z1)
    d2f1 = relu_deriv2(Z1)
    J_1 = ((W2 * W2).T).dot(d2Z2)
    J_2 = (W2.T).dot(dZ2)
    for row in range(0, m):
        theta = (J_1[row] * (df1[row] ** 2)) + (J_2[row] * d2f1[row])
        grad_W1_r = grad_W1_func_r(theta, X, Xi)
        H_r = H_matirx1_r(theta, X, Xi, grad_W1_r)
        L_r = L_matrix1_r(X, P, theta, dW1, db1, row, Xi, grad_W1_r)
        P_r = L_r.dot(np.linalg.pinv(H_r))
        if row == 0 :
            matrix1 = np.vstack([P_r])
        else :
            matrix1 = np.vstack([matrix1, P_r])
    cpW1 = np.delete(matrix1, n, axis=1)
    cpb1 = matrix1[:,n].reshape(m,1)

    i = np.ones((1, A1.shape[1]))
    A1i = np.vstack((A1, i))
    P = np.hstack([W2, b2])
    for row in range(0,p):
        grad_W2_r = grad_W2_func_r(A1, row, d2Z2, A1i)
        H_r = H_matirx2_r(A1, row, d2Z2, A1i, grad_W2_r)
        L_r = L_matrix2_r(A1, P, row, dW2, db2, d2Z2, A1i, grad_W2_r)
        P_r = L_r.dot(np.linalg.pinv(H_r))
        if row == 0 :
            matrix2 = np.vstack([P_r])
        else :
            matrix2 = np.vstack([matrix2, P_r])
    cpW2 = np.delete(matrix2, m, axis=1)
    cpb2 = matrix2[:,m].reshape(p,1)
    return cpW1, cpb1, cpW2, cpb2

def ret_weight(X, Y, W1, b1, W2, b2, iter=1) :
    cpW1 = W1.copy()
    cpb1 = b1.copy()
    cpW2 = W2.copy()
    cpb2 = b2.copy()
    for it in range(0, iter) :
        ###############
        y = Y.T
        Z1 = (X.T).dot(cpW1.T) + cpb1.T
        h = relu(Z1)
        Z2 = h.dot(cpW2.T) + cpb2
        y_pred = sigmoid(Z2)
        loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
        print(f'loss_Z-{it} : {loss}\n')
        if loss < 0.0005 :
            break
        cpW1, cpb1, cpW2, cpb2 = P_matrix(X, Y, cpW1, cpb1, cpW2, cpb2)
    return cpW1, cpb1, cpW2, cpb2



############################################################################################################################################
############################################################################################################################################
############################################################################################################################################



# 1. 데이터 생성 및 레이블 만들기
np.random.seed()
# 8차원 입력 데이터를 무작위 생성
X = np.random.randn(N, D)

true_w = np.random.randn(D)
true_b = 0.7

# 선형 결합 후 시그모이드로 확률 계산하여 이진 레이블 생성
logits = X.dot(true_w) + true_b
probabilities = 1 / (1 + np.exp(-logits))
y = (probabilities > 0.5).astype(np.float32).reshape(-1, 1)  # shape: (N, 1)

# 3. 모델 파라미터 초기화 및 학습 하이퍼파라미터 설정
# 입력 -> 은닉층 가중치와 bias (W1: (D, hidden_dim), b1: (1, hidden_dim))
l = 0.5
W1 = np.random.randn(D, hidden_dim) * l
b1 = np.random.randn(1, hidden_dim) * l


# 은닉층 -> 출력층 가중치와 bias (W2: (hidden_dim, 1), b2: (1, 1))
W2 = np.random.randn(hidden_dim, 1) * l
b2 = np.random.randn(1, 1) * l


##########################################################################################################
start = time.perf_counter()
###########################################################################################################
Z1_ = X.dot(W1) + b1         # (N, hidden_dim)
A1_ = relu(Z1_)               # (N, hidden_dim)
Z2_ = A1_.dot(W2) + b2        # (N, 1)
y_pred = sigmoid(Z2_)        # (N, 1)
loss0 = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
_W1, _b1, _W2, _b2 = ret_weight(X.T, y.T, W1.T, b1.T, W2.T, b2.T, iter=iterator)
Z1_ = X.dot(_W1.T) + _b1.T
h_ = relu(Z1_)
Z2_ = h_.dot(_W2.T) + _b2
y_pred_ = sigmoid(Z2_)
loss_ = -np.mean(y * np.log(y_pred_ + 1e-8) + (1 - y) * np.log(1 - y_pred_ + 1e-8))
# print(f'W1 :\n{_W1}\nb1 :\n{_b1}\nW2 :\n{_W2}\nb2 :\n{_b2}')
print(f'loss : {loss_}')
###########################################################################################################
end = time.perf_counter()
print("나의 코드 실행 시간: {:.4f} 초".format(end - start))
###########################################################################################################


# 3. Adam 하이퍼파라미터 설정
lr = 0.03
epochs = 1600
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Adam 모멘텀 변수 초기화 (모든 파라미터와 동일한 shape)
mW1 = np.zeros_like(W1)
vb1 = np.zeros_like(b1)
mW2 = np.zeros_like(W2)
vb2 = np.zeros_like(b2)
vW1 = np.zeros_like(W1)
vb1_v = np.zeros_like(b1)  # 이름을 다르게 해서 혼동 방지: b1의 2차 모멘트
vW2 = np.zeros_like(W2)
vb2_v = np.zeros_like(b2)

loss_history = []

start = time.perf_counter()
for epoch in range(1, epochs+1):
    # 순전파: 은닉층
    Z1 = X.dot(W1) + b1         # (N, hidden_dim)
    A1 = relu(Z1)               # (N, hidden_dim)
    # 순전파: 출력층
    Z2 = A1.dot(W2) + b2        # (N, 1)
    y_pred = sigmoid(Z2)        # (N, 1)
    
    # 손실 함수: 이진 크로스 엔트로피
    loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
    loss_history.append(loss)
    
    # 역전파: 출력층
    dZ2 = y_pred - y            # (N, 1)
    dW2 = A1.T.dot(dZ2) / N      # (hidden_dim, 1)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / N  # (1, 1)
    
    # 역전파: 은닉층
    dA1 = dZ2.dot(W2.T)         # (N, hidden_dim)
    dZ1 = dA1 * relu_deriv(Z1)  # (N, hidden_dim)
    dW1 = X.T.dot(dZ1) / N       # (D, hidden_dim)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / N  # (1, hidden_dim)
    
    # Adam 업데이트 (시간 단계 t = epoch)
    t = epoch
    
    # 1차 모멘트 업데이트
    mW1 = beta1 * mW1 + (1 - beta1) * dW1
    mW2 = beta1 * mW2 + (1 - beta1) * dW2
    vb1 = beta1 * vb1 + (1 - beta1) * db1
    vb2 = beta1 * vb2 + (1 - beta1) * db2
    
    # 2차 모멘트 업데이트
    vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
    vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
    vb1_v = beta2 * vb1_v + (1 - beta2) * (db1 ** 2)
    vb2_v = beta2 * vb2_v + (1 - beta2) * (db2 ** 2)
    
    # 편향 보정 (Bias correction)
    mW1_corr = mW1 / (1 - beta1 ** t)
    mW2_corr = mW2 / (1 - beta1 ** t)
    vb1_corr = vb1 / (1 - beta1 ** t)
    vb2_corr = vb2 / (1 - beta1 ** t)
    
    vW1_corr = vW1 / (1 - beta2 ** t)
    vW2_corr = vW2 / (1 - beta2 ** t)
    vb1_v_corr = vb1_v / (1 - beta2 ** t)
    vb2_v_corr = vb2_v / (1 - beta2 ** t)
    
    # 파라미터 업데이트
    W1 -= lr * mW1_corr / (np.sqrt(vW1_corr) + epsilon)
    b1 -= lr * vb1_corr / (np.sqrt(vb1_v_corr) + epsilon)
    W2 -= lr * mW2_corr / (np.sqrt(vW2_corr) + epsilon)
    b2 -= lr * vb2_corr / (np.sqrt(vb2_v_corr) + epsilon)
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

end = time.perf_counter()
print("Adam 학습 코드 실행 시간: {:.4f} 초".format(end - start))
print("학습 완료")
# print("학습된 W1:\n", W1)
# print("학습된 b1:\n", b1)
# print("학습된 W2:\n", W2)
# print("학습된 b2:\n", b2)


N_val = int(N * 0.5)
X_val = np.random.randn(N_val, D)
logits_val = X_val.dot(true_w) + true_b
probabilities_val = 1 / (1 + np.exp(-logits_val))
y_val = (probabilities_val > 0.5).astype(np.float32).reshape(-1, 1)

# 검증 데이터에 대해 순전파 수행
Z1_val = X_val.dot(W1) + b1      # (N_val, hidden_dim)
A1_val = relu(Z1_val)            # (N_val, hidden_dim)
Z2_val = A1_val.dot(W2) + b2     # (N_val, 1)
y_pred_val = sigmoid(Z2_val)     # (N_val, 1)

# 검증 손실 계산 (이진 크로스 엔트로피)
val_loss = -np.mean(y_val * np.log(y_pred_val + 1e-8) + (1 - y_val) * np.log(1 - y_pred_val + 1e-8))
print("Validation Loss: {:.4f}".format(val_loss))

Z1_val = X_val.dot(_W1.T) + _b1.T
A1_val = relu(Z1_val)
Z2_val = A1_val.dot(_W2.T) + _b2
y_pred_val = sigmoid(Z2_val)
loss_ = -np.mean(y_val * np.log(y_pred_val + 1e-8) + (1 - y_val) * np.log(1 - y_pred_val + 1e-8))
print(f'Validatio loss in my model : {loss_}')