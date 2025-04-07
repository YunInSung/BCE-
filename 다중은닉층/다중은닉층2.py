import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import time

N = 10000          # 데이터 샘플 수
D = 12             # 입력 차원
num_classes = 6   # 클래스 수
hidden_dim1 = 24    # 첫 번째 은닉층 크기
hidden_dim2 = hidden_dim1     # 두 번째 은닉층 크기
size = N
epsilon = 1e-8
iterator = 50
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
#############  W3 b3  ###############
#####################################


def grad_W3_func_r(A2, row, d2Z3, A2i):
    new_A1 = d2Z3[row] * A2 / N
    result = new_A1[:,np.newaxis,:] * A2i[np.newaxis,:,:]
    matrix = np.sum(result, axis=2).T
    return matrix

def H_matirx3_r(A2, row, d2Z3, A2i, grad_W3_r):
    m = A2.shape[0]
    matrix = np.hstack([grad_W3_r, np.sum(d2Z3[row] * A2i, axis = 1).reshape(m+1,1) / N])
    return matrix

def L_matrix3_r(A2, P, row, dW3, db3, d2Z3, A2i, grad_W3_r, learn):
    m = A2.shape[0]
    v0 = P[row]
    W3_r =  dW3[row] * learn
    matrix = (v0@grad_W3_r) - W3_r
    grad_b_r = np.sum(d2Z3[row] * A2i, axis = 1).reshape(m+1,1) / N
    b_r = db3[row] * learn
    tmp = v0.reshape(1, m + 1).dot(grad_b_r) - b_r
    matrix = np.append(matrix, tmp).reshape(1, m + 1)
    return matrix

#####################################
#############  W2 b2  ###############
#####################################


def grad_W2_func_r(theta, A1, A1i):
    tmp = theta * A1 / N
    result = tmp[:,np.newaxis,:] * A1i[np.newaxis,:,:]
    matrix = np.sum(result, axis=2).T
    return matrix

def H_matirx2_r(theta, A1, A1i, grad_W2_r):
    n = A1.shape[0]
    matrix = np.hstack([grad_W2_r, np.sum(theta * A1i, axis = 1).reshape(n + 1, 1) / N])
    return matrix

def L_matrix2_r(A1, P, theta, dW2, db2, row, A1i, grad_W2_r, learn):
    n = A1.shape[0]
    v0 = P[row]
    W2_r =  dW2[row] * learn
    matrix = (v0.dot(grad_W2_r)) - W2_r
    grad_b_r = np.sum(theta * A1i, axis = 1).reshape(n + 1, 1) / N
    b_r = db2[row] * learn
    tmp = v0.reshape(1, n + 1).dot(grad_b_r) - b_r
    matrix = np.append(matrix, tmp).reshape(1, n + 1)
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

#####################################
#############    P    ###############
#####################################

def P_matrix(X, Y, W1, b1, W2, b2, W3, b3, learn):
    matrix3 = 0
    matrix2 = 0
    matrix1 = 0
    p = Y.shape[0]
    m = W2.shape[0]
    n = W1.shape[0]
    r = X.shape[0]
    cpW1 = W1.copy()
    cpb1 = b1.copy()
    cpW2 = W2.copy()
    cpb2 = b2.copy()
    cpW3 = W3.copy()
    cpb3 = b3.copy()

    y = Y
    Z1 = W1@X + b1
    A1 = relu(Z1)
    Z2 = W2@A1 + b2
    A2 = relu(Z2)
    Z3 = W3@A2 + b3
    y_pred = (softmax(Z3.T)).T

    # 역전파: 출력층
    dZ3 = (y_pred - y)
    d2Z3 = (y_pred - y_pred * y_pred)
    dW3 = dZ3.dot(A2.T) / N
    db3 = np.sum(dZ3, axis=1, keepdims=True) / N
    # 역전파: 두 번째 은닉층
    dA2 = W3.T@dZ3
    dZ2 = dA2 * relu_deriv(Z2)
    dW2 = dZ2.dot(A1.T) / N
    db2 = np.sum(dZ2, axis=1, keepdims=True) / N
    # 역전파: 첫 번째 은닉층
    dA1 = W2.T@dZ2
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = dZ1.dot(X.T) / N
    db1 = np.sum(dZ1, axis=1, keepdims=True) / N

    # # ######################
    # loss_ = -np.mean(np.sum(y.T * np.log(y_pred.T + 1e-8), axis=1))
    # print(f'loss_test in : {loss_}')
    # # ######################

    i = np.ones((1, X.shape[1]))
    Xi = np.vstack((X, i))
    P = np.hstack([W1, b1])
    df1_Z1 = relu_deriv(Z1)
    d2f1_Z1 = relu_deriv2(Z1)
    df1_Z2 = relu_deriv(Z2)
    d2f1_Z2 = relu_deriv2(Z2)

    V = np.einsum('as,sp,sk->apk', W2.T, W3.T, df1_Z2)  # shape: (n, p, l)
    T1 = np.einsum('pk,apk->ak', y_pred, V**2)  # shape: (n, l)
    W3_y = np.dot(W3.T, y_pred)  # shape: (m, l)
    U = np.einsum('as,sk,sk->ak', W2.T, df1_Z2, W3_y)  # shape: (n, l)
    T2 = U**2
    tensor = T1 - T2

    J_1 = tensor + (W2.T ** 2).dot(((W3.T).dot(dZ3)) * d2f1_Z2)
    J_2 = (W2.T).dot(((W3.T).dot(dZ3)) * df1_Z2)
    for row in range(0, n):
        theta = (J_1[row] * (df1_Z1[row] ** 2)) + (J_2[row] * d2f1_Z1[row])
        grad_W1_r = grad_W1_func_r(theta, X, Xi)
        H_r = H_matirx1_r(theta, X, Xi, grad_W1_r)
        L_r = L_matrix1_r(X, P, theta, dW1, db1, row, Xi, grad_W1_r, learn)
        P_r = L_r.dot(np.linalg.pinv(H_r))
        if row == 0 :
            matrix1 = np.vstack([P_r])
        else :
            matrix1 = np.vstack([matrix1, P_r])
    cpW1 = np.delete(matrix1, r, axis=1)
    cpb1 = matrix1[:,r].reshape(n,1)

    i = np.ones((1, A1.shape[1]))
    A1i = np.vstack((A1, i))
    P = np.hstack([W2, b2])
    df1 = relu_deriv(Z2)
    d2f1 = relu_deriv2(Z2)
    term1 = np.dot(W3.T**2, y_pred)
    term2 = (np.dot(W3.T, y_pred))**2
    J_1 = term1 - term2
    # J_1 = np.dot(W3.T**2, d2Z3)
    J_2 = (W3.T).dot(dZ3)
    for row in range(0, m):
        theta = (J_1[row] * (df1[row] ** 2)) + (J_2[row] * d2f1[row])
        grad_W2_r = grad_W2_func_r(theta, A1, A1i)
        H_r = H_matirx2_r(theta, A1, A1i, grad_W2_r)
        L_r = L_matrix2_r(A1, P, theta, dW2, db2, row, A1i, grad_W2_r, learn)
        P_r = L_r.dot(np.linalg.pinv(H_r))
        if row == 0 :
            matrix2 = np.vstack([P_r])
        else :
            matrix2 = np.vstack([matrix2, P_r])
    cpW2 = np.delete(matrix2, n, axis=1)
    cpb2 = matrix2[:,n].reshape(m,1)

    i = np.ones((1, A2.shape[1]))
    A2i = np.vstack((A2, i))
    P = np.hstack([W3, b3])
    for row in range(0,p):
        grad_W3_r = grad_W3_func_r(A2, row, d2Z3, A2i)
        H_r = H_matirx3_r(A2, row, d2Z3, A2i, grad_W3_r)
        L_r = L_matrix3_r(A2, P, row, dW3, db3, d2Z3, A2i, grad_W3_r, learn)
        P_r = L_r.dot(np.linalg.pinv(H_r))
        if row == 0 :
            matrix3 = np.vstack([P_r])
        else :
            matrix3 = np.vstack([matrix3, P_r])
    cpW3 = np.delete(matrix3, m, axis=1)
    cpb3 = matrix3[:,m].reshape(p,1)
    return cpW1, cpb1, cpW2, cpb2, cpW3, cpb3

def ret_weight(X, Y, W1, b1, W2, b2, W3, b3, loss0, iter=1) :
    prev_loss = loss0
    loss = 1
    cpW1 = W1.copy()
    cpb1 = b1.copy()
    cpW2 = W2.copy()
    cpb2 = b2.copy()
    cpW3 = W3.copy()
    cpb3 = b3.copy()
    prevW1, prevb1, prevW2, prevb2, prevW3, prevb3 = cpW1, cpb1, cpW2, cpb2, cpW3, cpb3
    learn = 0.0025
    continous = 0
    for it in range(0, iter) :
        ###############
        if loss < 1e-2:
            break
        cpW1, cpb1, cpW2, cpb2, cpW3, cpb3 = P_matrix(X, Y, prevW1, prevb1, prevW2, prevb2, prevW3, prevb3, learn)
        continous += 1
        y = Y.T
        Z1 = (X.T).dot(cpW1.T) + cpb1.T         # (N, hidden_dim1)
        A1 = relu(Z1)
        Z2 = A1.dot(cpW2.T) + cpb2.T        # (N, hidden_dim2)
        A2 = relu(Z2)
        Z3 = A2.dot(cpW3.T) + cpb3.T        # (N, num_classes)
        y_pred_ = softmax(Z3)
        loss = -np.mean(np.sum(y * np.log(y_pred_ + 1e-8), axis=1))
        print(f'loss_Z-{it} : {loss}\n')
        if loss > prev_loss :
            learn *= 0.25
            continous = 0
            print(f'it : {it} - learn : {learn}')
            continue
        if learn < 0.2 :
            learn *= 2
            if learn > 0.2 :
                learn = 0.2
        prevW1, prevb1, prevW2, prevb2, prevW3, prevb3 = cpW1, cpb1, cpW2, cpb2, cpW3, cpb3
        prev_loss = loss
    return prevW1, prevb1, prevW2, prevb2, prevW3, prevb3



############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


# 1. 데이터 생성 및 레이블 만들기
# for i in [35, 49, 72, 82, 85, 104, 106, 115, 129, 137] : 
# for i in [137] : 
# for i in range(151, 301) : 
np.random.seed(82)
# 8차원 입력 데이터를 무작위 생성
# 고정된 centers 배열을 정의하여 학습 및 검증 데이터에 동일하게 적용합니다.
X, y = make_classification(n_samples=N,
                           n_features=D,
                           n_informative=D,   # 모든 특성이 정보를 가지도록
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=num_classes,
                           n_clusters_per_class=1,  # 각 클래스당 하나의 클러스터
                           flip_y=0,          # 라벨 노이즈 없음
                           class_sep=2.0,     # 클래스 간 분리 정도
                           random_state=42)

# 필요하면 다시 numpy 배열로 변환할 수 있습니다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
X = X_train
y = y_train.reshape(-1, 1)


# 원-핫 인코딩 함수
def one_hot(y, num_classes):
    onehot = np.zeros((y.shape[0], num_classes))
    onehot[np.arange(y.shape[0]), y.flatten()] = 1
    return onehot

y_onehot = one_hot(y, num_classes)

W1 = np.random.randn(D, hidden_dim1) * np.sqrt(2.0 / D)
b1 = np.zeros((1, hidden_dim1))
W2 = np.random.randn(hidden_dim1, hidden_dim2) * np.sqrt(2.0 / hidden_dim1)
b2 = np.zeros((1, hidden_dim2))
W3 = np.random.randn(hidden_dim2, num_classes) * np.sqrt(2.0 / hidden_dim2)
b3 = np.zeros((1, num_classes))


Z1_ = X.dot(W1) + b1         # (N, hidden_dim1)
A1_ = relu(Z1_)
# 순전파: 두 번째 은닉층
Z2_ = A1_.dot(W2) + b2        # (N, hidden_dim2)
A2_ = relu(Z2_)
# 순전파: 출력층
Z3_ = A2_.dot(W3) + b3        # (N, num_classes)
y_pred_ = softmax(Z3_)
# 손실 함수: 범주형 교차 엔트로피
loss0 = -np.mean(np.sum(y_onehot * np.log(y_pred_ + 1e-8), axis=1))
print(f'loss0 = {loss0}')


Z1_ = (W1.T).dot(X.T) + b1.T        # (N, hidden_dim1)
A1_ = relu(Z1_)
# 순전파: 두 번째 은닉층
Z2_ = (W2.T).dot(A1_) + b2.T        # (N, hidden_dim2)
A2_ = relu(Z2_)
# 순전파: 출력층
Z3_ = (W3.T).dot(A2_) + b3.T        # (N, num_classes)
y_pred_ = (softmax(Z3_.T)).T
loss_ = -np.mean(np.sum(y_onehot * np.log(y_pred_.T + 1e-8), axis=1))
print(f'loss_test : {loss_}')
##########################################################################################################
##########################################################################################################
##########################################################################################################
start = time.perf_counter()
###########################################################################################################
_W1, _b1, _W2, _b2, _W3, _b3 = ret_weight(X.T, y_onehot.T, W1.T, b1.T, W2.T, b2.T, W3.T, b3.T, loss0, iter=iterator)
###########################################################################################################
end = time.perf_counter()
print("나의 코드 실행 시간: {:.4f} 초".format(end - start))
###########################################################################################################
Z1_ = X.dot(_W1.T) + _b1.T         # (N, hidden_dim1)
A1_ = relu(Z1_)
Z2_ = A1_.dot(_W2.T) + _b2.T        # (N, hidden_dim2)
A2_ = relu(Z2_)
Z3_ = A2_.dot(_W3.T) + _b3.T        # (N, num_classes)
y_pred_ = softmax(Z3_)
loss_ = -np.mean(np.sum(y_onehot * np.log(y_pred_ + 1e-8), axis=1))
# print(f'W1 :\n{_W1}\nb1 :\n{_b1}\nW2 :\n{_W2}\nb2 :\n{_b2}')
print("loss : : {:.6f}".format(loss_))
##########################################################################################################
##########################################################################################################


# Adam 하이퍼파라미터 설정
lr = 0.05
epochs = 600
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
    if loss < 1e-2:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
        break
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
X_val = X_test
y_val = y_test.reshape(-1, 1)
y_val_onehot = one_hot(y_val, num_classes)

Z1_val = X_val.dot(W1) + b1
A1_val = relu(Z1_val)
Z2_val = A1_val.dot(W2) + b2
A2_val = relu(Z2_val)
Z3_val = A2_val.dot(W3) + b3
y_pred_val = softmax(Z3_val)
val_loss = -np.mean(np.sum(y_val_onehot * np.log(y_pred_val + 1e-8), axis=1))
print("Validation Loss: {:.6f}".format(val_loss))



Z1_val = X_val.dot(_W1.T) + _b1.T
A1_val = relu(Z1_val)
Z2_val = A1_val.dot(_W2.T) + _b2.T
A2_val = relu(Z2_val)
Z3_val = A2_val.dot(_W3.T) + _b3.T
y_pred_val = softmax(Z3_val)
loss_ = -np.mean(np.sum(y_val_onehot * np.log(y_pred_val + 1e-8), axis=1))
print("Validatio loss in my model : {:.6f}".format(loss_))
print(f'data : {N}')