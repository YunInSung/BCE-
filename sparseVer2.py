from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import tensorflow as tf
import time


beta1 = 0.9
beta2 = 0.999
adam_epsilon = 1e-8

###############################################################################################################
###############################################################################################################
###############################################################################################################
  
# fetch dataset 
statlog_shuttle = fetch_ucirepo(id=148) 
  
# data (as pandas dataframes) 
X = statlog_shuttle.data.features 
y = statlog_shuttle.data.targets

# 3. 학습/테스트 데이터 분할 (예: 80% 학습, 20% 테스트)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 4. 특성 스케일링 (StandardScaler를 사용하여 평균 0, 분산 1로 변환)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 레이블 One-Hot 인코딩 (Covertype 데이터는 다중 분류 문제)
# y_train, y_test가 pandas Series인 경우 .values를 이용해 numpy 배열로 변환
encoder = OneHotEncoder(sparse_output=False)  # scikit-learn 1.2 이상에서는 sparse_output 사용
y_train_onehot = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.values.reshape(-1, 1))

###############################################################################################################
###############################################################################################################
###############################################################################################################
# 상수 및 데이터셋 설정
N = X_train_scaled.shape[0]         # 데이터 샘플 수
D = X_train_scaled.shape[1]            # 입력 차원
hidden_dim = 21
iterator = 75
num_classes = y_train_onehot.shape[1]  # 클래스 수
epsilon = 1e-16
N_float = tf.cast(N, tf.float32)

# mL = tf.Variable(tf.zeros(shape=(hidden_dim, D + 1)), trainable=False)
# vL = tf.Variable(tf.zeros(shape=(hidden_dim, D + 1)), trainable=False)
# mL2 = tf.Variable(tf.zeros(shape=(num_classes, hidden_dim + 1)), trainable=False)
# vL2 = tf.Variable(tf.zeros(shape=(num_classes, hidden_dim + 1)), trainable=False)
###############################################################################################################
###############################################################################################################
###############################################################################################################

# NumPy 데이터 -> TensorFlow 텐서 (dtype=tf.float32)
X_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)           # (N, D)
y_onehot_tf = tf.convert_to_tensor(y_train_onehot, dtype=tf.float32)  # (N, num_classes)

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

# --- 최적화를 위한 tf.function 데코레이터 적용 ---

@tf.function
def grad_W2_func_r_tf_batch(h, row, d2Z2, A1i):
    """
    h:     공통 입력 텐서, shape = (80, 5000)
    d2Z2:  전체 d2Z2 텐서, shape = (p, 5000)
    A1i:   보조 입력 텐서, shape = (81, 5000)
    row:   배치 인덱스, shape = (B,)
    
    출력:
      각 배치에 대해 계산한 결과, shape = (B, 81, 80)
      (원래 단일 케이스에서는 (81,80)이므로, 배치 차원이 추가된 버전)
    """
    # d2Z2_batch: (B, 5000)
    d2Z2_batch = tf.gather(d2Z2, row)
    # new_h: broadcast하여 (B,80,5000)
    # h: (80,5000) → (1,80,5000)
    new_h = (tf.expand_dims(d2Z2_batch, axis=1) * h) / N_float  # (B,80,5000)
    # new_h_exp: (B,80,1,5000)
    new_h_exp = tf.expand_dims(new_h, axis=2)
    # A1i_exp: A1i: (81,5000) → (1,81,5000) → (1,1,81,5000)
    A1i_exp = tf.expand_dims(tf.expand_dims(A1i, axis=0), axis=1)
    # 곱: (B,80,1,5000) * (1,1,81,5000) → (B,80,81,5000)
    res = new_h_exp * A1i_exp  
    # reduce sum over M axis (axis=-1): (B,80,81)
    res_sum = tf.reduce_sum(res, axis=-1)
    # transpose 각 배치: (B,80,81) → (B,81,80)
    matrix = tf.transpose(res_sum, perm=[0, 2, 1])
    return matrix

@tf.function
def H_matirx2_r_tf_batch(h, row, d2Z2, A1i, grad_W2_r):
    """
    벡터화된 연산을 사용하여, 각 배치에 대해 H 행렬을 계산.
    A1i: (81,5000) → out_dim = 81.
    grad_W2_r: (B,81,80)
    
    출력: (B, 81, 81)
    """
    B = tf.shape(row)[0]
    out_dim = tf.shape(A1i)[0]  # 81
    d2Z2_batch = tf.gather(d2Z2, row)  # (B,5000)
    # A1i: (81,5000) → (1,81,5000)
    A1i_exp = tf.expand_dims(A1i, axis=0)
    # d2Z2_batch: (B,5000) → (B,1,5000)
    d2Z2_exp = tf.expand_dims(d2Z2_batch, axis=1)
    prod = d2Z2_exp * A1i_exp          # (B,81,5000)
    summed = tf.reduce_sum(prod, axis=2)  # (B,81)
    sum_tensor = tf.reshape(summed / N_float, [B, out_dim, 1])  # (B,81,1)
    matrix = tf.concat([grad_W2_r, sum_tensor], axis=2)  # (B,81,80+1) -> (B,81,81)
    return matrix


@tf.function
def grad_W1_func_r_tf(theta, X, Xi):
    """
    theta: (B, M)
    X:     (D, M)
    Xi:    (D+1, M)
    
    출력: (B, D+1, D)
    """
    X_b = tf.expand_dims(X, 0)       # (1,D,M)
    Xi_b = tf.expand_dims(Xi, 0)     # (1,D+1,M)
    theta_exp = tf.expand_dims(theta, 1)  # (B,1,M)
    tmp = theta_exp * X_b / N_float       # (B,D,M)
    tmp_exp = tf.expand_dims(tmp, 2)        # (B,D,1,M)
    Xi_exp = tf.expand_dims(Xi_b, 1)          # (B,1,D+1,M)
    result = tmp_exp * Xi_exp               # (B,D,D+1,M)
    result_sum = tf.reduce_sum(result, axis=-1)  # (B,D,D+1)
    matrix = tf.transpose(result_sum, perm=[0, 2, 1])  # (B,D+1,D)
    return matrix

@tf.function
def H_matirx1_r_tf(theta, X, Xi, grad_W1_r):
    """
    theta: (B,M), X: (D,M), Xi: (D+1,M), grad_W1_r: (B,D+1,D)
    출력: (B, D+1, D+1)
    """
    theta_exp = tf.expand_dims(theta, 1)  # (B,1,M)
    Xi_exp = tf.expand_dims(Xi, 0)          # (1, D+1, M)
    prod = theta_exp * Xi_exp               # (B, D+1, M)
    sum_tensor = tf.reduce_sum(prod, axis=-1)  # (B,D+1)
    sum_tensor = tf.reshape(sum_tensor / N_float, [tf.shape(theta)[0], -1, 1])  # (B,D+1,1)
    matrix = tf.concat([grad_W1_r, sum_tensor], axis=2)  # (B, D+1, D+1)
    return matrix

@tf.function
def P_matrix_tf(X, Y, W1, b1, W2, b2, learn, it):
    """
    X: (D,N), Y: (p,N), W1: (m,D), b1: (m,1), W2: (p,m), b2: (p,1)
    출력: cpW1: (m,D), cpb1: (m,1), cpW2: (p,m), cpb2: (p,1)
    """
    p = tf.shape(Y)[0]      # num_classes
    m = tf.shape(W1)[0]     # hidden_dim
    n = tf.shape(X)[0]      # 입력 차원 D

    Z1 = tf.matmul(W1, X) + b1         # (m,N)
    A1 = tf.nn.leaky_relu(Z1, alpha=0.001)
    Z2 = tf.matmul(W2, A1) + b2         # (p,N)
    y_pred = tf.transpose(tf.nn.softmax(tf.transpose(Z2)))
    dZ2 = y_pred - Y
    d2Z2 = y_pred - tf.square(y_pred)
    dW2 = tf.matmul(dZ2, tf.transpose(A1)) / tf.cast(N, tf.float32)
    db2 = tf.reduce_sum(dZ2, axis=1, keepdims=True) / tf.cast(N, tf.float32)
    dA1 = tf.matmul(tf.transpose(W2), dZ2)
    Z1_deriv = tf.where(Z1>=0, tf.ones_like(Z1), 0.001*tf.ones_like(Z1))
    dZ1 = dA1 * Z1_deriv
    dW1 = tf.matmul(dZ1, tf.transpose(X)) / tf.cast(N, tf.float32)
    db1 = tf.reduce_sum(dZ1, axis=1, keepdims=True) / tf.cast(N, tf.float32)

    #######################################################################################
    #################################      W1 b1      #####################################
    #######################################################################################
    i = tf.ones((1, tf.shape(X)[1]), dtype=tf.float32)
    Xi = tf.concat([X, i], axis=0)   # (D+1,N)
    P1 = tf.concat([W1, b1], axis=1)  # (m,D+1)
    GP1 = tf.concat([dW1, db1], axis=1)
    df1 = tf.where(Z1>=0, tf.ones_like(Z1), 0.001*tf.ones_like(Z1))
    d2f1 = tf.zeros_like(Z1)

    hidden_rows = tf.range(m)
    term1 = tf.matmul(tf.transpose(W2)**2, y_pred) 
    term2 = tf.square(tf.matmul(tf.transpose(W2), y_pred))
    J_1 = term1 - term2
    theta = (tf.gather(J_1, hidden_rows) * tf.square(df1) +
             tf.gather(tf.matmul(tf.transpose(W2), dZ2), hidden_rows) * d2f1)
    grad_W1_r = grad_W1_func_r_tf(theta, X, Xi)        # (m, D+1, D)
    H_r       = H_matirx1_r_tf(theta, X, Xi, grad_W1_r)  # (m, D+1, D+1)
    H_r_reg   = H_r + epsilon * tf.eye(tf.shape(H_r)[1], dtype=tf.float32)
    sol       = tf.linalg.solve(tf.transpose(H_r_reg, perm=[0,2,1]), tf.expand_dims(GP1, axis=2))
    L_r       = tf.squeeze(tf.transpose(sol, perm=[0, 2, 1]), axis=1)

    # mL.assign(beta1 * mL + (1 - beta1) * L_r)
    # vL.assign(beta2 * vL + (1 - beta2) * tf.square(L_r))
    # mL_corr = mL / (1 - beta1**it)
    # vL_corr = vL / (1 - beta2**it)
    # matrix1 = P1 - learn * mL_corr / (tf.sqrt(vL_corr) + adam_epsilon)

    matrix1 = P1 - learn * L_r
    cpW1 = matrix1[:, :n]         
    cpb1 = tf.reshape(matrix1[:, n], [m, 1])
    #######################################################################################
    #################################      W2 b2      #####################################
    #######################################################################################
    i_A1 = tf.ones((1, tf.shape(A1)[1]), dtype=tf.float32)
    A1i = tf.concat([A1, i_A1], axis=0)  # (m+1,N)
    P2 = tf.concat([W2, b2], axis=1)      # (p, m+1)
    GP2 = tf.concat([dW2, db2], axis=1)

    output_rows = tf.range(p)
    grad_W2_r = grad_W2_func_r_tf_batch(A1, output_rows, d2Z2, A1i)
    H_r2 = H_matirx2_r_tf_batch(A1, output_rows, d2Z2, A1i, grad_W2_r)
    H_r2_reg = H_r2 + epsilon * tf.eye(tf.shape(H_r2)[1], dtype=tf.float32)
    sol2 = tf.linalg.solve(tf.transpose(H_r2_reg, perm=[0,2,1]), tf.expand_dims(GP2, axis=2))
    L_r2 = tf.squeeze(tf.transpose(sol2, perm=[0, 2, 1]), axis=1)

    # mL2.assign(beta1 * mL2 + (1 - beta1) * L_r2)
    # vL2.assign(beta2 * vL2 + (1 - beta2) * tf.square(L_r2))
    # mL2_corr = mL2 / (1 - beta1**it)
    # vL2_corr = vL2 / (1 - beta2**it)
    # matrix2 = P2 - learn * mL2_corr / (tf.sqrt(vL2_corr) + adam_epsilon)

    matrix2 = P2 - learn * L_r2
    cpW2 = matrix2[:, :m]
    cpb2 = tf.reshape(matrix2[:, m], [p, 1])
    return cpW1, cpb1, cpW2, cpb2

@tf.function
def ret_weight_tf(X, Y, W1, b1, W2, b2, loss0, iter=1):
    prev_loss = loss0
    loss = tf.constant(1.0, dtype=tf.float32)
    cpW1 = tf.identity(W1)
    cpb1 = tf.identity(b1)
    cpW2 = tf.identity(W2)
    cpb2 = tf.identity(b2)
    prevW1, prevb1, prevW2, prevb2 = cpW1, cpb1, cpW2, cpb2
    # prevmL, prevvL, prevmL2, prevvL2 = tf.identity(mL), tf.identity(vL), tf.identity(mL2), tf.identity(vL2)
    learn = 0.05
    continuous = 0
    y_T = tf.transpose(Y)

    for it in tf.range(iter):
        if loss < 1e-2 * 1.5:
            break
        cpW1, cpb1, cpW2, cpb2 = P_matrix_tf(X, Y, prevW1, prevb1, prevW2, prevb2, learn, tf.cast(it+1, tf.float32))
        continuous += 1
        Z1 = tf.matmul(cpW1, X) + cpb1
        h = tf.nn.leaky_relu(Z1, alpha=0.001)
        Z2 = tf.matmul(cpW2, h) + cpb2
        y_pred = tf.nn.softmax(tf.transpose(Z2))
        loss = -tf.reduce_mean(tf.reduce_sum(y_T * tf.math.log(y_pred + 1e-8), axis=1))
        tf.print("loss_Z-", it, ":", loss)
        if loss - prev_loss > 0:
            if (learn < 1e-6) :
                break
            learn = learn * 0.25
            continuous = 0
            # mL.assign(prevmL)
            # vL.assign(prevvL)
            # mL2.assign(prevmL2)
            # vL2.assign(prevvL2)
            tf.print("it:", it, "- learn:", learn)
            continue
        if continuous >= 2 and learn < 0.1:
            learn = tf.minimum(learn * 1.5, 0.1)
        prevW1, prevb1, prevW2, prevb2 = cpW1, cpb1, cpW2, cpb2
        # prevmL, prevvL, prevmL2, prevvL2 = tf.identity(mL), tf.identity(vL), tf.identity(mL2), tf.identity(vL2)
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

W1_init_copy = W1_init.copy()
b1_init_copy = b1_init.copy()
W2_init_copy = W2_init.copy()
b2_init_copy = b2_init.copy()

W1_tf_var_copy = tf.Variable(W1_init_copy, dtype=tf.float32)
b1_tf_var_copy = tf.Variable(b1_init_copy, dtype=tf.float32)
W2_tf_var_copy = tf.Variable(W2_init_copy, dtype=tf.float32)
b2_tf_var_copy = tf.Variable(b2_init_copy, dtype=tf.float32)

##############################################
#                Adam 학습                #
##############################################
lr = 0.25
epochs = 800

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
    A1 = tf.nn.leaky_relu(Z1, alpha=0.001)
    Z2 = tf.matmul(A1, W2_tf_var) + b2_tf_var
    y_pred = tf.nn.softmax(Z2)
    loss = -tf.reduce_mean(tf.reduce_sum(y_onehot_tf * tf.math.log(y_pred + 1e-8), axis=1))
    loss_history.append(loss.numpy())

    if loss < 0.015:
        tf.print("Epoch", epoch, "Loss:", loss)
        break
    
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


# 초기 순전파 및 손실 계산 (NumPy와 동일한 방식)
Z1 = tf.matmul(X_tf, W1_tf_var_copy) + b1_tf_var_copy
A1 = tf.nn.leaky_relu(Z1, alpha=0.001)
Z2 = tf.matmul(A1, W2_tf_var_copy) + b2_tf_var_copy
y_pred = tf.nn.softmax(Z2)
loss0_tf = -tf.reduce_mean(tf.reduce_sum(y_onehot_tf * tf.math.log(y_pred + 1e-8), axis=1))
tf.print("loss0 =", loss0_tf)

##############################################
#        ret_weight 함수 실행 (예시)         #
##############################################
start = time.perf_counter()
# ret_weight_tf의 입력은 전치된 텐서들이어야 함.
_W1_tf, _b1_tf, _W2_tf, _b2_tf = ret_weight_tf(tf.transpose(X_tf), tf.transpose(y_onehot_tf),
                                                tf.transpose(W1_tf_var_copy), tf.transpose(b1_tf_var_copy),
                                                tf.transpose(W2_tf_var_copy), tf.transpose(b2_tf_var_copy),
                                                loss0_tf, iter=iterator)
end = time.perf_counter()
print("ret_weight 코드 실행 시간: {:.4f} 초".format(end - start))

Z1_val = tf.matmul(X_tf, tf.transpose(_W1_tf)) + tf.transpose(_b1_tf)
A1_val = tf.nn.leaky_relu(Z1_val, alpha=0.001)
Z2_val = tf.matmul(A1_val, tf.transpose(_W2_tf)) + tf.transpose(_b2_tf)
y_pred_val = tf.nn.softmax(Z2_val)
loss_val = -tf.reduce_mean(tf.reduce_sum(y_onehot_tf * tf.math.log(y_pred_val + 1e-8), axis=1))
tf.print("my loss :", loss_val)


X_val_tf = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
y_val_onehot_tf = tf.convert_to_tensor(y_test_onehot, dtype=tf.float32)

Z1_val = tf.matmul(X_val_tf, W1_tf_var) + b1_tf_var
A1_val = tf.nn.leaky_relu(Z1_val, alpha=0.001)
Z2_val = tf.matmul(A1_val, W2_tf_var) + b2_tf_var
y_pred_val = tf.nn.softmax(Z2_val)
val_loss = -tf.reduce_mean(tf.reduce_sum(y_val_onehot_tf * tf.math.log(y_pred_val + 1e-8), axis=1))
tf.print("Validation Loss:", val_loss)

Z1_val = tf.matmul(X_val_tf, tf.transpose(_W1_tf)) + tf.transpose(_b1_tf)
A1_val = tf.nn.leaky_relu(Z1_val, alpha=0.001)
Z2_val = tf.matmul(A1_val, tf.transpose(_W2_tf)) + tf.transpose(_b2_tf)
y_pred_val = tf.nn.softmax(Z2_val)
loss_val = -tf.reduce_mean(tf.reduce_sum(y_val_onehot_tf * tf.math.log(y_pred_val + 1e-8), axis=1))
tf.print("my validation loss :", loss_val)

  
