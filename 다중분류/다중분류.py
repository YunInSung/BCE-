import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time

N = 10000         # 데이터 샘플 수
D = 12             # 입력 차원
hidden_dim = 24
size = N
epsilon = 1e-8
iterator = 50
num_classes = 6   # 클래스 수
######################################################################################################
######################################################################################################
######################################################################################################
# np.seterr(over='raise', under='raise', divide='raise', invalid='raise')
# 소프트맥스 함수 (수치안정을 위해 각 행의 최댓값을 빼줌)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x, alpha=0.001):
    return np.where(x >= 0, x, alpha * x)

def relu_deriv(x, alpha=0.001):
    return np.where(x >= 0, 1, alpha)

def relu_deriv2(x, alpha=0.001):
    return np.zeros_like(x)

# def relu(x, alpha=0.001):
#     return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

# def relu_deriv(x, alpha=0.001):
#     # x = 0에서는 1로 정의 (일반적인 구현)
#     return np.where(x >= 0, 1, alpha * np.exp(x))

# def relu_deriv2(x, alpha=0.001):
#     # x >= 0에서는 0, x < 0에서는 alpha * exp(x)
#     return np.where(x >= 0, 0, alpha * np.exp(x))

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

def L_matrix2_r(h, P, row, dW2, db2, d2Z2, A1i, grad_W2_r, learn):
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

def L_matrix1_r(X, P, theta, dW1, db1, row, Xi, grad_W1_r, learn):
    n = X.shape[0]
    v0 = P[row]
    W1_r =  dW1[row] * learn
    matrix = (v0.dot(grad_W1_r)) - W1_r
    grad_b_r = np.sum(theta * Xi, axis = 1).reshape(n + 1, 1) / N
    b_r = db1[row] * learn
    tmp = v0.reshape(1, n + 1).dot(grad_b_r) - b_r
    matrix = np.append(matrix, tmp).reshape(1, n + 1)
    return matrix

def P_matrix(X, Y, W1, b1, W2, b2, learn):
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
    y_pred = (softmax(Z2.T)).T
    dZ2 = (y_pred - y)            # (N, 1)
    d2Z2 = (y_pred - y_pred * y_pred)
    dW2 = dZ2.dot(A1.T) / N     # (hidden_dim, 1)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / N  # (1, 1)
    # 은닉층: dL/dA1 = dZ2 dot W2^T, dL/dZ1 = dL/dA1 * ReLU'(Z1)
    dA1 = W2.T@dZ2         # (N, hidden_dim)
    dZ1 = dA1 * relu_deriv(Z1)  # (N, hidden_dim)
    dW1 = dZ1.dot(X.T) / N      # (D, hidden_dim)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / N  # (1, hidden_dim)

    # # ######################
    # loss_ = -np.mean(np.sum(y.T * np.log(y_pred.T + 1e-8), axis=1))
    # print(f'loss_test in : {loss_}')
    # # ######################

    i = np.ones((1, X.shape[1]))
    Xi = np.vstack((X, i))
    P = np.hstack([W1, b1])
    df1 = relu_deriv(Z1)
    d2f1 = relu_deriv2(Z1)
    # J_1 = np.dot(W2.T**2, d2Z2)
    term1 = np.dot(W2.T**2, y_pred)
    term2 = (np.dot(W2.T, y_pred))**2
    J_1 = term1 - term2
    J_2 = (W2.T).dot(dZ2)
    for row in range(0, m):
        theta = (J_1[row] * (df1[row] ** 2)) + (J_2[row] * d2f1[row])
        grad_W1_r = grad_W1_func_r(theta, X, Xi)
        H_r = H_matirx1_r(theta, X, Xi, grad_W1_r)
        L_r = L_matrix1_r(X, P, theta, dW1, db1, row, Xi, grad_W1_r, learn)
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
        L_r = L_matrix2_r(A1, P, row, dW2, db2, d2Z2, A1i, grad_W2_r, learn)
        P_r = L_r.dot(np.linalg.pinv(H_r))
        if row == 0 :
            matrix2 = np.vstack([P_r])
        else :
            matrix2 = np.vstack([matrix2, P_r])
    cpW2 = np.delete(matrix2, m, axis=1)
    cpb2 = matrix2[:,m].reshape(p,1)
    return cpW1, cpb1, cpW2, cpb2

def ret_weight(X, Y, W1, b1, W2, b2, loss0, iter=1) :
    prev_loss = loss0
    loss = 1
    cpW1 = W1.copy()
    cpb1 = b1.copy()
    cpW2 = W2.copy()
    cpb2 = b2.copy()
    prevW1, prevb1, prevW2, prevb2 = cpW1, cpb1, cpW2, cpb2
    learn = 0.01
    continous = 0
    for it in range(0, iter) :
        ###############
        if loss < 1e-4:
            break
        cpW1, cpb1, cpW2, cpb2 = P_matrix(X, Y, prevW1, prevb1, prevW2, prevb2, learn)
        continous += 1
        y = Y.T
        Z1 = (X.T).dot(cpW1.T) + cpb1.T
        h = relu(Z1)
        Z2 = h.dot(cpW2.T) + cpb2.T
        y_pred = softmax(Z2)
        loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
        print(f'loss_Z-{it} : {loss}\n')
        if loss > prev_loss :
            learn *= 0.5
            continous = 0
            print(f'it : {it} - learn : {learn}')
            continue
        if continous >= 2 and learn < 0.75 :
            learn *= 1.35
        prevW1, prevb1, prevW2, prevb2 = cpW1, cpb1, cpW2, cpb2
        prev_loss = loss
    return prevW1, prevb1, prevW2, prevb2



############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


# 1. 데이터 생성 및 레이블 만들기
# for i in [35, 49, 72, 82, 85, 104, 106, 115, 129, 137] : 
# for i in [137] : 
# for i in range(151, 301) : 
np.random.seed(129)
# 8차원 입력 데이터를 무작위 생성
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

W1 = np.random.randn(D, hidden_dim) * np.sqrt(2.0 / D)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim)
b2 = np.zeros((1, num_classes))


Z1_ = X.dot(W1) + b1         # (N, hidden_dim)
A1_ = relu(Z1_)               # (N, hidden_dim)
Z2_ = A1_.dot(W2) + b2        # (N, num_classes)
y_pred = softmax(Z2_)        # (N, num_classes)
loss0 = -np.mean(np.sum(y_onehot * np.log(y_pred + 1e-8), axis=1))
print(f'loss0 = {loss0}')

Z1_ = (W1.T).dot(X.T) + b1.T
h_ = relu(Z1_)
Z2_ = (W2.T).dot(h_) + b2.T
y_pred_ = (softmax(Z2_.T)).T
loss_ = -np.mean(np.sum(y_onehot * np.log(y_pred_.T + 1e-8), axis=1))
# print(f'W1 :\n{_W1}\nb1 :\n{_b1}\nW2 :\n{_W2}\nb2 :\n{_b2}')
print(f'loss_test : {loss_}')
##########################################################################################################
##########################################################################################################
##########################################################################################################
start = time.perf_counter()
###########################################################################################################
_W1, _b1, _W2, _b2 = ret_weight(X.T, y_onehot.T, W1.T, b1.T, W2.T, b2.T, loss0, iter=iterator)
###########################################################################################################
end = time.perf_counter()
print("나의 코드 실행 시간: {:.4f} 초".format(end - start))
###########################################################################################################
Z1_ = X.dot(_W1.T) + _b1.T
h_ = relu(Z1_)
Z2_ = h_.dot(_W2.T) + _b2.T
y_pred_ = softmax(Z2_)
loss_ = -np.mean(np.sum(y_onehot * np.log(y_pred_ + 1e-8), axis=1))
# print(f'W1 :\n{_W1}\nb1 :\n{_b1}\nW2 :\n{_W2}\nb2 :\n{_b2}')
print("loss : : {:.6f}".format(loss_))
##########################################################################################################
##########################################################################################################


# 3. Adam 하이퍼파라미터 설정
lr = 0.25
epochs = 800
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

    if epoch == 1 or epoch == 50 :
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    if epoch % 200 == 0 or epoch == epochs:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

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
print("Validation Loss: {:.6f}".format(val_loss))


Z1_val = X_val.dot(_W1.T) + _b1.T
A1_val = relu(Z1_val)
Z2_val = A1_val.dot(_W2.T) + _b2.T
y_pred_val = softmax(Z2_val)
loss_ = -np.mean(np.sum(y_val_onehot * np.log(y_pred_val + 1e-8), axis=1))
print("Validatio loss in my model : {:.6f}".format(loss_))
print(f'data : {N}')