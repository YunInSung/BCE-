import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time

# 상수 및 데이터셋 설정
N = 10000         # 데이터 샘플 수
D = 40            # 입력 차원
hidden_dim = 80
iterator = 50
num_classes = 20  # 클래스 수
epsilon = 1e-16
N_float = tf.cast(N, tf.float32)

# 데이터 생성 (NumPy)
np.random.seed()
centers = np.array([
    np.full(D, -5.0, dtype=np.float32),
    np.full(D, 0.0, dtype=np.float32),
    np.full(D, 5.0, dtype=np.float32)
])
X_np, y_np = make_blobs(n_samples=N, n_features=D, centers=centers, 
                          cluster_std=1.5, random_state=42)
X_np = X_np.astype(np.float32)
y_np = y_np.reshape(-1, 1)

def one_hot(y, num_classes):
    onehot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    onehot[np.arange(y.shape[0]), y.flatten()] = 1
    return onehot

y_onehot_np = one_hot(y_np, num_classes)

# NumPy 데이터 -> TensorFlow 텐서 (dtype=tf.float32)
X_tf = tf.convert_to_tensor(X_np, dtype=tf.float32)           # (N, D)
y_tf = tf.convert_to_tensor(y_np, dtype=tf.float32)            # (N, 1)
y_onehot_tf = tf.convert_to_tensor(y_onehot_np, dtype=tf.float32)  # (N, num_classes)

# 활성화 및 미분 함수 정의
def softmax(x):
    x_max = tf.reduce_max(x, axis=1, keepdims=True)
    exp_x = tf.exp(x - x_max)
    return exp_x / tf.reduce_sum(exp_x, axis=1, keepdims=True)

def relu(x, alpha=0.001):
    return tf.where(x >= 0, x, alpha * x)

def relu_deriv(x, alpha=0.001):
    return tf.where(x >= 0, tf.ones_like(x), tf.fill(tf.shape(x), alpha))

def relu_deriv2(x, alpha=0.001):
    return tf.zeros_like(x)

##############################################
#       TensorFlow 버전의 W2, b2 관련 함수      #
##############################################
def grad_W2_func_r_tf(h, row, d2Z2, A1i):
    new_h = (d2Z2[row] * h) / N_float
    result = tf.expand_dims(new_h, axis=1) * tf.expand_dims(A1i, axis=0)
    matrix = tf.transpose(tf.reduce_sum(result, axis=2))
    return matrix

def H_matirx2_r_tf(h, row, d2Z2, A1i, grad_W2_r):
    m = tf.shape(h)[0]
    sum_tensor = tf.reduce_sum(d2Z2[row] * A1i, axis=1)
    sum_tensor = tf.reshape(sum_tensor / N_float, [m+1, 1])
    matrix = tf.concat([grad_W2_r, sum_tensor], axis=1)
    return matrix

def L_matrix2_r_tf(h, P, row, dW2, db2, d2Z2, A1i, grad_W2_r, learn):
    m = tf.shape(h)[0]
    v0 = P[row]
    W2_r = dW2[row] * learn
    matrix = tf.matmul(tf.reshape(v0, [1, -1]), grad_W2_r)
    matrix = tf.reshape(matrix, [-1]) - W2_r
    grad_b_r = tf.reshape(tf.reduce_sum(d2Z2[row] * A1i, axis=1), [m+1, 1]) / N_float
    b_r = db2[row] * learn
    tmp = tf.matmul(tf.reshape(v0, [1, -1]), grad_b_r) - b_r
    matrix = tf.concat([tf.reshape(matrix, [-1]), tf.reshape(tmp, [-1])], axis=0)
    matrix = tf.reshape(matrix, [1, m+1])
    return matrix

##############################################
#       TensorFlow 버전의 W1, b1 관련 함수      #
##############################################
def grad_W1_func_r_tf(theta, X, Xi):
    tmp = theta * X / N_float
    result = tf.expand_dims(tmp, axis=1) * tf.expand_dims(Xi, axis=0)
    matrix = tf.transpose(tf.reduce_sum(result, axis=2))
    return matrix

def H_matirx1_r_tf(theta, X, Xi, grad_W1_r):
    n = tf.shape(X)[0]  # 입력 차원 D
    sum_tensor = tf.reduce_sum(theta * Xi, axis=1)
    sum_tensor = tf.reshape(sum_tensor / N_float, [n+1, 1])
    matrix = tf.concat([grad_W1_r, sum_tensor], axis=1)
    return matrix

def L_matrix1_r_tf(X, P, theta, dW1, db1, row, Xi, grad_W1_r, learn):
    n = tf.shape(X)[0]  # D
    v0 = P[row]
    W1_r = dW1[row] * learn
    matrix = tf.matmul(tf.reshape(v0, [1, -1]), grad_W1_r)
    matrix = tf.reshape(matrix, [-1]) - W1_r
    grad_b_r = tf.reshape(tf.reduce_sum(theta * Xi, axis=1), [n+1, 1]) / N_float
    b_r = db1[row] * learn
    tmp = tf.matmul(tf.reshape(v0, [1, -1]), grad_b_r) - b_r
    matrix = tf.concat([tf.reshape(matrix, [-1]), tf.reshape(tmp, [-1])], axis=0)
    matrix = tf.reshape(matrix, [1, n+1])
    return matrix

##############################################
#        P_matrix 및 ret_weight 함수        #
##############################################
def P_matrix_tf(X, Y, W1, b1, W2, b2, learn):
    # X: (D, N), Y: (p, N) with p = num_classes, W1: (m, D), b1: (m,1),
    # W2: (p, m), b2: (p,1)
    p = tf.shape(Y)[0]      # num_classes
    m = tf.shape(W1)[0]     # 은닉층 노드 수 (hidden_dim)
    n = tf.shape(X)[0]      # 입력 차원 D

    Z1 = tf.matmul(W1, X) + b1         # (m, N)
    A1 = relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2         # (p, N)
    y_pred = softmax(tf.transpose(Z2))  # (N, p)
    y_pred = tf.transpose(y_pred)       # (p, N)
    dZ2 = y_pred - Y
    d2Z2 = y_pred - y_pred * y_pred
    dW2 = tf.matmul(dZ2, tf.transpose(A1)) / tf.cast(N, tf.float32)
    db2 = tf.reduce_sum(dZ2, axis=1, keepdims=True) / tf.cast(N, tf.float32)
    dA1 = tf.matmul(tf.transpose(W2), dZ2)
    Z1_deriv = relu_deriv(Z1)
    dZ1 = dA1 * Z1_deriv
    dW1 = tf.matmul(dZ1, tf.transpose(X)) / tf.cast(N, tf.float32)
    db1 = tf.reduce_sum(dZ1, axis=1, keepdims=True) / tf.cast(N, tf.float32)

    i = tf.ones((1, tf.shape(X)[1]), dtype=tf.float32)
    Xi = tf.concat([X, i], axis=0)   # (D+1, N)
    P1 = tf.concat([W1, b1], axis=1)  # (m, D+1)
    df1 = relu_deriv(Z1)
    d2f1 = relu_deriv2(Z1)
    # J_1와 J_2 계산
    J_1 = tf.matmul(tf.transpose(W2)**2, d2Z2)   # (m, N)
    J_2 = tf.matmul(tf.transpose(W2), dZ2)         # (m, N)

    # 은닉층 업데이트: 각 hidden row별 계산을 tf.map_fn으로 처리
    def process_hidden_row(row_idx):
        theta = (J_1[row_idx] * (df1[row_idx] ** 2)) + (J_2[row_idx] * d2f1[row_idx])
        # start = time.perf_counter()
        grad_W1_r = grad_W1_func_r_tf(theta, X, Xi)    # (D+1, D)
        H_r = H_matirx1_r_tf(theta, X, Xi, grad_W1_r)    # (D+1, D+1)
        L_r = L_matrix1_r_tf(X, P1, theta, dW1, db1, row_idx, Xi, grad_W1_r, learn)  # (1, D+1)
        # end = time.perf_counter()
        # print("역행렬1 코드 실행 시간: {:.4f} 초".format(end - start))
        H_r_reg = H_r + epsilon * tf.eye(tf.shape(H_r)[0], dtype=tf.float32)
        P_r = tf.transpose(tf.linalg.solve(tf.transpose(H_r_reg), tf.transpose(L_r)))
        return P_r  # 예상 shape: (1, D+1)

    matrix1 = tf.map_fn(process_hidden_row, tf.range(m), dtype=tf.float32)
    # matrix1 shape가 (m, 1, D+1)라면 squeeze를 사용합니다.
    matrix1 = tf.squeeze(matrix1, axis=1)  # 결과 shape: (m, D+1)
    cpW1 = matrix1[:, :n]         # (m, D)
    cpb1 = tf.reshape(matrix1[:, n], [m, 1])  # (m, 1)

    i_A1 = tf.ones((1, tf.shape(A1)[1]), dtype=tf.float32)
    A1i = tf.concat([A1, i_A1], axis=0)  # (m+1, N)
    P2 = tf.concat([W2, b2], axis=1)      # (p, m+1)
    # 출력층 업데이트: 각 output row별 계산을 tf.map_fn으로 처리
    def process_output_row(row_idx):
        # start = time.perf_counter()
        grad_W2_r = grad_W2_func_r_tf(A1, row_idx, d2Z2, A1i)  # (m+1, m)
        H_r = H_matirx2_r_tf(A1, row_idx, d2Z2, A1i, grad_W2_r)  # (m+1, m+1)
        L_r = L_matrix2_r_tf(A1, P2, row_idx, dW2, db2, d2Z2, A1i, grad_W2_r, learn)  # (1, m+1)
        # end = time.perf_counter()
        # print("역행렬2 코드 실행 시간: {:.4f} 초".format(end - start))
        H_r_reg = H_r + epsilon * tf.eye(tf.shape(H_r)[0], dtype=tf.float32)
        P_r = tf.transpose(tf.linalg.solve(tf.transpose(H_r_reg), tf.transpose(L_r)))
        return P_r  # 예상 shape: (1, m+1)

    matrix2 = tf.map_fn(process_output_row, tf.range(p), dtype=tf.float32)
    matrix2 = tf.squeeze(matrix2, axis=1)  # 결과 shape: (p, m+1)
    cpW2 = matrix2[:, :m]         # (p, m)
    cpb2 = tf.reshape(matrix2[:, m], [p, 1])  # (p, 1)
    return cpW1, cpb1, cpW2, cpb2


def ret_weight_tf(X, Y, W1, b1, W2, b2, loss0, iter=1):
    prev_loss = loss0
    loss = tf.constant(1.0, dtype=tf.float32)
    cpW1 = tf.identity(W1)
    cpb1 = tf.identity(b1)
    cpW2 = tf.identity(W2)
    cpb2 = tf.identity(b2)
    prevW1, prevb1, prevW2, prevb2 = cpW1, cpb1, cpW2, cpb2
    learn = 0.01
    continuous = 0
    for it in range(iter):
        if loss < 1e-4:
            break
        cpW1, cpb1, cpW2, cpb2 = P_matrix_tf(X, Y, prevW1, prevb1, prevW2, prevb2, learn)
        continuous += 1
        y_T = tf.transpose(Y)
        Z1 = tf.matmul(tf.transpose(X), tf.transpose(cpW1)) + tf.transpose(cpb1)
        h = relu(Z1)
        Z2 = tf.matmul(h, tf.transpose(cpW2)) + tf.transpose(cpb2)
        y_pred = softmax(Z2)
        loss = -tf.reduce_mean(tf.reduce_sum(y_T * tf.math.log(y_pred + 1e-8), axis=1))
        tf.print("loss_Z-", it, ":", loss)
        if loss > prev_loss:
            learn *= 0.5
            continuous = 0
            tf.print("it:", it, "- learn:", learn)
            continue
        if continuous >= 3 and learn < 0.25:
            learn *= 3
            if learn > 0.25 :
              learn = 0.25
        prevW1, prevb1, prevW2, prevb2 = cpW1, cpb1, cpW2, cpb2
        prev_loss = loss
    return prevW1, prevb1, prevW2, prevb2

##############################################
#         초기 가중치 및 모델 평가           #
##############################################
W1_init = np.random.randn(D, hidden_dim).astype(np.float32) * np.sqrt(2.0 / D)
b1_init = np.zeros((1, hidden_dim), dtype=np.float32)
W2_init = np.random.randn(hidden_dim, num_classes).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
b2_init = np.zeros((1, num_classes), dtype=np.float32)

W1_tf_var = tf.Variable(W1_init, dtype=tf.float32)
b1_tf_var = tf.Variable(b1_init, dtype=tf.float32)
W2_tf_var = tf.Variable(W2_init, dtype=tf.float32)
b2_tf_var = tf.Variable(b2_init, dtype=tf.float32)

# 초기 순전파 및 손실 계산 (NumPy와 동일한 방식)
Z1 = tf.matmul(X_tf, W1_tf_var) + b1_tf_var
A1 = relu(Z1)
Z2 = tf.matmul(A1, W2_tf_var) + b2_tf_var
y_pred = softmax(Z2)
loss0_tf = -tf.reduce_mean(tf.reduce_sum(y_onehot_tf * tf.math.log(y_pred + 1e-8), axis=1))
tf.print("loss0 =", loss0_tf)

##############################################
#        ret_weight 함수 실행 (예시)         #
##############################################
start = time.perf_counter()
# ret_weight_tf의 입력은 전치된 텐서들이어야 함.
_W1_tf, _b1_tf, _W2_tf, _b2_tf = ret_weight_tf(tf.transpose(X_tf), tf.transpose(y_onehot_tf),
                                                tf.transpose(W1_tf_var), tf.transpose(b1_tf_var),
                                                tf.transpose(W2_tf_var), tf.transpose(b2_tf_var),
                                                loss0_tf, iter=iterator)
end = time.perf_counter()
print("ret_weight 코드 실행 시간: {:.4f} 초".format(end - start))

Z1_val = tf.matmul(X_tf, tf.transpose(_W1_tf)) + tf.transpose(_b1_tf)
A1_val = relu(Z1_val)
Z2_val = tf.matmul(A1_val, tf.transpose(_W2_tf)) + tf.transpose(_b2_tf)
y_pred_val = softmax(Z2_val)
loss_val = -tf.reduce_mean(tf.reduce_sum(y_onehot_tf * tf.math.log(y_pred_val + 1e-8), axis=1))
tf.print("최종 loss :", loss_val)

##############################################
#                Adam 학습                #
##############################################
lr = 0.25
epochs = 800
beta1 = 0.9
beta2 = 0.999
adam_epsilon = 1e-8

mW1 = tf.Variable(tf.zeros_like(W1_tf_var), trainable=False)
vb1 = tf.Variable(tf.zeros_like(b1_tf_var), trainable=False)
mW2 = tf.Variable(tf.zeros_like(W2_tf_var), trainable=False)
vb2 = tf.Variable(tf.zeros_like(b2_tf_var), trainable=False)
vW1 = tf.Variable(tf.zeros_like(W1_tf_var), trainable=False)
vW2 = tf.Variable(tf.zeros_like(W2_tf_var), trainable=False)
vb1_v = tf.Variable(tf.zeros_like(b1_tf_var), trainable=False)
vb2_v = tf.Variable(tf.zeros_like(b2_tf_var), trainable=False)

loss_history = []
start = time.perf_counter()
for epoch in range(1, epochs+1):
    Z1 = tf.matmul(X_tf, W1_tf_var) + b1_tf_var
    A1 = relu(Z1)
    Z2 = tf.matmul(A1, W2_tf_var) + b2_tf_var
    y_pred = softmax(Z2)
    loss = -tf.reduce_mean(tf.reduce_sum(y_onehot_tf * tf.math.log(y_pred + 1e-8), axis=1))
    loss_history.append(loss.numpy())
    
    dZ2 = y_pred - y_onehot_tf
    dW2 = tf.matmul(tf.transpose(A1), dZ2) / tf.cast(N, tf.float32)
    db2_grad = tf.reduce_sum(dZ2, axis=0, keepdims=True) / tf.cast(N, tf.float32)
    
    dA1 = tf.matmul(dZ2, tf.transpose(W2_tf_var))
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = tf.matmul(tf.transpose(X_tf), dZ1) / tf.cast(N, tf.float32)
    db1_grad = tf.reduce_sum(dZ1, axis=0, keepdims=True) / tf.cast(N, tf.float32)
    
    t = epoch
    mW1.assign(beta1 * mW1 + (1 - beta1) * dW1)
    mW2.assign(beta1 * mW2 + (1 - beta1) * dW2)
    vb1.assign(beta1 * vb1 + (1 - beta1) * db1_grad)
    vb2.assign(beta1 * vb2 + (1 - beta1) * db2_grad)
    
    vW1.assign(beta2 * vW1 + (1 - beta2) * tf.square(dW1))
    vW2.assign(beta2 * vW2 + (1 - beta2) * tf.square(dW2))
    vb1_v.assign(beta2 * vb1_v + (1 - beta2) * tf.square(db1_grad))
    vb2_v.assign(beta2 * vb2_v + (1 - beta2) * tf.square(db2_grad))
    
    mW1_corr = mW1 / (1 - beta1**t)
    mW2_corr = mW2 / (1 - beta1**t)
    vb1_corr = vb1 / (1 - beta1**t)
    vb2_corr = vb2 / (1 - beta1**t)
    vW1_corr = vW1 / (1 - beta2**t)
    vW2_corr = vW2 / (1 - beta2**t)
    vb1_v_corr = vb1_v / (1 - beta2**t)
    vb2_v_corr = vb2_v / (1 - beta2**t)
    
    W1_tf_var.assign(W1_tf_var - lr * mW1_corr / (tf.sqrt(vW1_corr) + adam_epsilon))
    b1_tf_var.assign(b1_tf_var - lr * vb1_corr / (tf.sqrt(vb1_v_corr) + adam_epsilon))
    W2_tf_var.assign(W2_tf_var - lr * mW2_corr / (tf.sqrt(vW2_corr) + adam_epsilon))
    b2_tf_var.assign(b2_tf_var - lr * vb2_corr / (tf.sqrt(vb2_v_corr) + adam_epsilon))
    
    if epoch in [1, 50]:
        tf.print("Epoch", epoch, "Loss:", loss)
    if epoch % 200 == 0 or epoch == epochs:
        tf.print("Epoch", epoch, "Loss:", loss)
end = time.perf_counter()
print("Adam 학습 코드 실행 시간: {:.4f} 초".format(end - start))
print("학습 완료")

N_val = int(N * 0.5)
X_val_np, y_val_np = make_blobs(n_samples=N_val, n_features=D, centers=centers, cluster_std=1.5, random_state=42)
X_val_np = X_val_np.astype(np.float32)
y_val_np = y_val_np.reshape(-1, 1)
y_val_onehot_np = one_hot(y_val_np, num_classes)
X_val_tf = tf.convert_to_tensor(X_val_np, dtype=tf.float32)
y_val_onehot_tf = tf.convert_to_tensor(y_val_onehot_np, dtype=tf.float32)

Z1_val = tf.matmul(X_val_tf, W1_tf_var) + b1_tf_var
A1_val = relu(Z1_val)
Z2_val = tf.matmul(A1_val, W2_tf_var) + b2_tf_var
y_pred_val = softmax(Z2_val)
val_loss = -tf.reduce_mean(tf.reduce_sum(y_val_onehot_tf * tf.math.log(y_pred_val + 1e-8), axis=1))
tf.print("Validation Loss:", val_loss)

Z1_val = tf.matmul(X_val_tf, tf.transpose(_W1_tf)) + tf.transpose(_b1_tf)
A1_val = relu(Z1_val)
Z2_val = tf.matmul(A1_val, tf.transpose(_W2_tf)) + tf.transpose(_b2_tf)
y_pred_val = softmax(Z2_val)
loss_val = -tf.reduce_mean(tf.reduce_sum(y_val_onehot_tf * tf.math.log(y_pred_val + 1e-8), axis=1))
tf.print("my validation loss :", loss_val)
