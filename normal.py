# 현 데이터 말고도, 임의로 만들어낸 데이터로도 실험해보기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp
import time
import math
import os
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' 또는 '3'으로 설정하면 INFO 메시지를 숨깁니다.


beta1 = 0.9
beta2 = 0.999
adam_epsilon = 1e-8
split_num = 10
adam_mini_batch = 100


# 상수 및 데이터셋 설정
prop = 0.2
N = 25000         # 데이터 샘플 수
D = 6            # 입력 차원

###############################################################################################################
###############################################################################################################
###############################################################################################################
  
# 데이터 생성 (NumPy)
np.random.seed(42)
center_values = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
centers = np.array([np.full(D, val, dtype=np.float32) for val in center_values])
X_np, y_np = make_blobs(n_samples=N, n_features=D, centers=centers, 
                          cluster_std=1.5, random_state=42)
X_np = X_np.astype(np.float32)
y_np = y_np.reshape(-1, 1)  # (N, 1)

# 2. 학습/테스트 데이터 분할 (80% 학습, 20% 테스트)
# stratify는 1차원 배열을 요구하므로 ravel() 사용
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np.ravel(), test_size=prop, stratify=y_np.ravel(), random_state=42
)

_, X_var, _, y_var = train_test_split(
    X_np, y_np.ravel(), test_size=prop, stratify=y_np.ravel(), random_state=42
)

# 3. Stratified K-Fold 분할 (NumPy 배열 인덱싱 사용)
skf = StratifiedKFold(n_splits=split_num, shuffle=True, random_state=42)
folds = []
for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    # NumPy 배열 인덱싱 사용 (.iloc 대신)
    X_fold = X_train[val_index]
    y_fold = y_train[val_index]
    folds.append((X_fold, y_fold))
    # 클래스 분포 확인 (np.unique 사용)
    unique, counts = np.unique(y_fold, return_counts=True)

# 4. 특성 스케일링: RobustScaler 사용
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_var_scaled = scaler.transform(X_var)

# 5. 레이블 One-Hot 인코딩
# OneHotEncoder는 입력이 2D 배열이어야 하므로 reshape(-1,1) 사용
encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
y_var_onehot = encoder.transform(y_var.reshape(-1, 1))

# 전체 클래스 목록 설정 (필요한 경우)
all_classes = np.unique(y_train)
encoder = OneHotEncoder(categories=[all_classes], sparse_output=False)
encoder.fit(y_train.reshape(-1, 1))

# 6. 각 폴드에 대해 스케일링, One-Hot 인코딩 후 TensorFlow 텐서로 변환
transformed_folds_tensor = []
for X_fold, y_fold in folds:
    X_fold_scaled = scaler.transform(X_fold)
    y_fold_onehot = encoder.transform(y_fold.reshape(-1, 1))
    X_tf_fold = tf.convert_to_tensor(X_fold_scaled, dtype=tf.float32)
    y_tf_fold = tf.convert_to_tensor(y_fold_onehot, dtype=tf.float32)
    transformed_folds_tensor.append((X_tf_fold, y_tf_fold))

###################################################################################
###################################################################################
N = X_train_scaled.shape[0]
hidden_dim = int(D * 2)
num_classes = 6 # 클래스 수
iterator = 6
epsilon = 1e-16
N_float = tf.cast(N, tf.float32)
batch_size = int(N/adam_mini_batch)
# batch_size = 128
steps_per_epoch = math.ceil(N / batch_size)
###################################################################################
###################################################################################

# NumPy 데이터 -> TensorFlow 텐서 (학습 데이터: 80%의 데이터)
X_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)           # (N*0.8, D)
y_onehot_tf = tf.convert_to_tensor(y_train_onehot, dtype=tf.float32)      # (N*0.8, num_classes)
X_var__tf = tf.convert_to_tensor(X_var_scaled, dtype=tf.float32)
y_var__onehot_tf = tf.convert_to_tensor(y_var_onehot, dtype=tf.float32)

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

# @tf.function
def P_matrix_tf(X, Y, W1, b1, W2, b2, learn):
    y_T = tf.transpose(Y)
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

    prev_loss = -tf.reduce_mean(tf.reduce_sum(Y * tf.math.log(y_pred + 1e-8), axis=0))
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
    pinv_A = tf.linalg.pinv(tf.transpose(H_r, perm=[0,2,1]))
    sol = tf.matmul(pinv_A, tf.expand_dims(GP1, axis=2))
    L_r = tf.squeeze(tf.transpose(sol, perm=[0,2,1]), axis=1)

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
    pinv_A = tf.linalg.pinv(tf.transpose(H_r2, perm=[0,2,1]))
    sol2 = tf.matmul(pinv_A, tf.expand_dims(GP2, axis=2))
    L_r2 = tf.squeeze(tf.transpose(sol2, perm=[0, 2, 1]), axis=1)

    normL = np.linalg.norm(L_r)
    normL2 = np.linalg.norm(L_r2)
    loss = 0
    for _ in range(0,7) :
        # matrix1 = P1 - learn * L_r
        matrix1 = P1 - learn * (L_r / normL)
        cpW1 = matrix1[:, :n]         
        cpb1 = tf.reshape(matrix1[:, n], [m, 1])
        # matrix2 = P2 - learn * L_r2
        matrix2 = P2 - learn * (L_r2 / normL2)
        cpW2 = matrix2[:, :m]
        cpb2 = tf.reshape(matrix2[:, m], [p, 1])
        Z1 = tf.matmul(cpW1, X) + cpb1
        h = tf.nn.leaky_relu(Z1, alpha=0.001)
        Z2 = tf.matmul(cpW2, h) + cpb2
        y_pred = tf.nn.softmax(tf.transpose(Z2))
        loss = -tf.reduce_mean(tf.reduce_sum(y_T * tf.math.log(y_pred + 1e-8), axis=1))
        if loss < prev_loss :
            break
        else :
            learn *= 0.25
    return cpW1, cpb1, cpW2, cpb2, learn

# @tf.function
def ret_weight_tf(transformed_folds_tensor, W1, b1, W2, b2, iter=1):
    loss = tf.constant(1.0, dtype=tf.float32)
    cpW1 = tf.identity(W1)
    cpb1 = tf.identity(b1)
    cpW2 = tf.identity(W2)
    cpb2 = tf.identity(b2)
    num_folds = len(transformed_folds_tensor)

    prev_loss = 10
    prevW1, prevb1, prevW2, prevb2 = cpW1, cpb1, cpW2, cpb2
    X_T = tf.transpose(X_var__tf)
    for it in tf.range(iter):
        x_batch, y_batch = 0, 0
        shuffled_indices = np.random.permutation(num_folds)
        learn = tf.constant(3, dtype=tf.float32)
        for idx in shuffled_indices:
            x_batch, y_batch = transformed_folds_tensor[idx]
            X = tf.transpose(x_batch)
            Y = tf.transpose(y_batch)
            cpW1, cpb1, cpW2, cpb2, learn = P_matrix_tf(X, Y, cpW1, cpb1, cpW2, cpb2, learn)
            if learn < 1e-5:
                break
            if learn < 2.5:
                learn = tf.minimum(learn * 2, 2.5)
        Z1 = tf.matmul(cpW1, X_T) + cpb1
        h = tf.nn.leaky_relu(Z1, alpha=0.001)
        Z2 = tf.matmul(cpW2, h) + cpb2
        y_pred = tf.nn.softmax(tf.transpose(Z2))
        loss = -tf.reduce_mean(tf.reduce_sum(y_var__onehot_tf * tf.math.log(y_pred + 1e-8), axis=1))
        print(f'loss{it} : {loss}')
        if prev_loss < loss :
            cpW1, cpb1, cpW2, cpb2 = prevW1, prevb1, prevW2, prevb2
        else :
            prevW1, prevb1, prevW2, prevb2 = cpW1, cpb1, cpW2, cpb2
            prev_loss = loss
    return cpW1, cpb1, cpW2, cpb2

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

W1_init_copy2 = W1_init.copy()
b1_init_copy2 = b1_init.copy()
W2_init_copy2 = W2_init.copy()
b2_init_copy2 = b2_init.copy()

##############################################
#        ret_weight 함수 실행 (예시)         #
##############################################

# 초기 순전파 및 손실 계산 (NumPy와 동일한 방식)
Z1 = tf.matmul(X_tf, W1_tf_var_copy) + b1_tf_var_copy
A1 = tf.nn.leaky_relu(Z1, alpha=0.001)
Z2 = tf.matmul(A1, W2_tf_var_copy) + b2_tf_var_copy
y_pred = tf.nn.softmax(Z2)
loss0_tf = -tf.reduce_mean(tf.reduce_sum(y_onehot_tf * tf.math.log(y_pred + 1e-8), axis=1))
tf.print("loss0 =", loss0_tf)

start = time.perf_counter()
# ret_weight_tf의 입력은 전치된 텐서들이어야 함.
_W1_tf, _b1_tf, _W2_tf, _b2_tf = ret_weight_tf(transformed_folds_tensor,
                                                tf.transpose(W1_tf_var_copy), tf.transpose(b1_tf_var_copy),
                                                tf.transpose(W2_tf_var_copy), tf.transpose(b2_tf_var_copy),
                                                iter=iterator)
end = time.perf_counter()
print("ret_weight 코드 실행 시간: {:.4f} 초".format(end - start))

Z1_val = tf.matmul(X_tf, tf.transpose(_W1_tf)) + tf.transpose(_b1_tf)
A1_val = tf.nn.leaky_relu(Z1_val, alpha=0.001)
Z2_val = tf.matmul(A1_val, tf.transpose(_W2_tf)) + tf.transpose(_b2_tf)
y_pred_val = tf.nn.softmax(Z2_val)
loss_val = -tf.reduce_mean(tf.reduce_sum(y_onehot_tf * tf.math.log(y_pred_val + 1e-8), axis=1))
tf.print("my loss :", loss_val)

##############################################
#                Adam 학습                #
##############################################
lr = 0.05
epochs = 600

mW1 = tf.Variable(tf.zeros_like(W1_tf_var), trainable=False)
vb1 = tf.Variable(tf.zeros_like(b1_tf_var), trainable=False)
mW2 = tf.Variable(tf.zeros_like(W2_tf_var), trainable=False)
vb2 = tf.Variable(tf.zeros_like(b2_tf_var), trainable=False)
vW1 = tf.Variable(tf.zeros_like(W1_tf_var), trainable=False)
vW2 = tf.Variable(tf.zeros_like(W2_tf_var), trainable=False)
vb1_v = tf.Variable(tf.zeros_like(b1_tf_var), trainable=False)
vb2_v = tf.Variable(tf.zeros_like(b2_tf_var), trainable=False)

loss_history = []

# 데이터셋을 tf.data.Dataset 형태로 구성 (X_tf, y_onehot_tf는 전체 데이터의 텐서)
dataset = tf.data.Dataset.from_tensor_slices((X_tf, y_onehot_tf))
dataset = dataset.shuffle(buffer_size=N, reshuffle_each_iteration=True).batch(batch_size).repeat()

start = time.perf_counter()

for epoch in range(1, epochs+1):
    epoch_losses = []  # 에포크 내 미니배치 손실 저장 리스트
    
    # 한 에폭(epoch)은 전체 데이터셋을 순회하는 것
    for X_batch, y_batch in dataset.take(steps_per_epoch):
        # 미니배치 크기를 동적으로 구하기
        current_batch_size = tf.cast(tf.shape(X_batch)[0], tf.float32)
        
        # 순전파(forward pass)
        Z1 = tf.matmul(X_batch, W1_tf_var) + b1_tf_var
        A1 = tf.nn.leaky_relu(Z1, alpha=0.001)
        Z2 = tf.matmul(A1, W2_tf_var) + b2_tf_var
        y_pred = tf.nn.softmax(Z2)
        
        # 손실 함수 계산 (교차 엔트로피)
        loss = -tf.reduce_mean(tf.reduce_sum(y_batch * tf.math.log(y_pred + 1e-8), axis=1))
        epoch_losses.append(loss.numpy())
        
        # 역전파 (backward pass)
        dZ2 = y_pred - y_batch
        dW2 = tf.matmul(tf.transpose(A1), dZ2) / current_batch_size
        db2_grad = tf.reduce_sum(dZ2, axis=0, keepdims=True) / current_batch_size
        
        dA1 = tf.matmul(dZ2, tf.transpose(W2_tf_var))
        # relu_deriv 함수는 활성함수의 미분을 계산하는 함수라고 가정합니다.
        dZ1 = dA1 * relu_deriv(Z1)
        dW1 = tf.matmul(tf.transpose(X_batch), dZ1) / current_batch_size
        db1_grad = tf.reduce_sum(dZ1, axis=0, keepdims=True) / current_batch_size
        
        # 여기서는 단순화를 위해 에폭 번호를 t로 사용 (미니배치 업데이트 수 반영 X)
        t = epoch
        
        # 1차 모멘텀 업데이트
        mW1.assign(beta1 * mW1 + (1 - beta1) * dW1)
        mW2.assign(beta1 * mW2 + (1 - beta1) * dW2)
        vb1.assign(beta1 * vb1 + (1 - beta1) * db1_grad)
        vb2.assign(beta1 * vb2 + (1 - beta1) * db2_grad)
        
        # 2차 모멘텀 업데이트
        vW1.assign(beta2 * vW1 + (1 - beta2) * tf.square(dW1))
        vW2.assign(beta2 * vW2 + (1 - beta2) * tf.square(dW2))
        vb1_v.assign(beta2 * vb1_v + (1 - beta2) * tf.square(db1_grad))
        vb2_v.assign(beta2 * vb2_v + (1 - beta2) * tf.square(db2_grad))
        
        # Bias correction
        mW1_corr = mW1 / (1 - beta1**t)
        mW2_corr = mW2 / (1 - beta1**t)
        vb1_corr = vb1 / (1 - beta1**t)
        vb2_corr = vb2 / (1 - beta1**t)
        vW1_corr = vW1 / (1 - beta2**t)
        vW2_corr = vW2 / (1 - beta2**t)
        vb1_v_corr = vb1_v / (1 - beta2**t)
        vb2_v_corr = vb2_v / (1 - beta2**t)
        
        # 파라미터 업데이트
        W1_tf_var.assign(W1_tf_var - lr * mW1_corr / (tf.sqrt(vW1_corr) + adam_epsilon))
        b1_tf_var.assign(b1_tf_var - lr * vb1_corr / (tf.sqrt(vb1_v_corr) + adam_epsilon))
        W2_tf_var.assign(W2_tf_var - lr * mW2_corr / (tf.sqrt(vW2_corr) + adam_epsilon))
        b2_tf_var.assign(b2_tf_var - lr * vb2_corr / (tf.sqrt(vb2_v_corr) + adam_epsilon))
    
    # 에포크당 평균 손실 계산
    epoch_loss_avg = np.mean(epoch_losses)
    loss_history.append(epoch_loss_avg)

    if epoch_loss_avg < 0.02:
        tf.print("Epoch", epoch, "Loss:", epoch_loss_avg)
        break
    
    # 중간 결과 출력
    if epoch in [1, 50] or epoch % 200 == 0 or epoch == epochs:
        tf.print("Epoch", epoch, "Loss:", epoch_loss_avg)

end = time.perf_counter()
print("Adam 미니배치 학습 코드 실행 시간: {:.4f} 초".format(end - start))
print("학습 완료")



X_val_tf = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
y_val_onehot_tf = tf.convert_to_tensor(y_test_onehot, dtype=tf.float32)
y_true = tf.argmax(y_val_onehot_tf, axis=1).numpy()


Z1_val = tf.matmul(X_val_tf, W1_tf_var) + b1_tf_var
A1_val = tf.nn.leaky_relu(Z1_val, alpha=0.001)
Z2_val = tf.matmul(A1_val, W2_tf_var) + b2_tf_var
y_pred_val = tf.nn.softmax(Z2_val)
val_loss = -tf.reduce_mean(tf.reduce_sum(y_val_onehot_tf * tf.math.log(y_pred_val + 1e-8), axis=1))
tf.print("Validation Loss:", val_loss)
y_pred = tf.argmax(y_pred_val, axis=1).numpy()
# 정확도 계산
acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)
# 분류 리포트 출력
print(classification_report(y_true, y_pred))
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("adam Confusion Matrix")
# plt.show()

Z1_val = tf.matmul(X_val_tf, tf.transpose(_W1_tf)) + tf.transpose(_b1_tf)
A1_val = tf.nn.leaky_relu(Z1_val, alpha=0.001)
Z2_val = tf.matmul(A1_val, tf.transpose(_W2_tf)) + tf.transpose(_b2_tf)
y_pred_val = tf.nn.softmax(Z2_val)
loss_val = -tf.reduce_mean(tf.reduce_sum(y_val_onehot_tf * tf.math.log(y_pred_val + 1e-8), axis=1))
tf.print("my validation loss :", loss_val)
y_pred = tf.argmax(y_pred_val, axis=1).numpy()
# 정확도 계산
acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)
# 분류 리포트 출력
print(classification_report(y_true, y_pred))
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("my Confusion Matrix")
# plt.show()
  

# # 초기 변수는 tf.Variable로 선언할 수도 있지만, lbfgs_minimize에서는 초기 파라미터 벡터를 사용합니다.
# # 따라서 아래와 같이 초기 파라미터 벡터를 생성합니다.
# init_params = np.concatenate([
#     W1_init_copy2.ravel(), 
#     b1_init_copy2.ravel(), 
#     W2_init_copy2.ravel(), 
#     b2_init_copy2.ravel()
# ])

# # 전체 학습/테스트 데이터를 Tensor로 변환
# X_train_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
# y_train_tf = tf.convert_to_tensor(y_train_onehot, dtype=tf.float32)
# X_test_tf  = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
# y_test_tf  = tf.convert_to_tensor(y_test_onehot, dtype=tf.float32)

# # 미니배치 크기 설정 (예: 128)
# def neural_net_loss_minibatch(params):
#     """
#     파라미터 벡터를 재구성하고, 무작위 미니배치를 사용하여 
#     1-hidden layer 신경망의 cross-entropy loss를 계산합니다.
#     """
#     # 미니배치 샘플링
#     num_samples = tf.shape(X_train_tf)[0]
#     indices = tf.random.uniform(shape=[batch_size], minval=0, maxval=num_samples, dtype=tf.int32)
#     X_batch = tf.gather(X_train_tf, indices)
#     y_batch = tf.gather(y_train_tf, indices)
    
#     # 파라미터 벡터에서 각 변수 추출
#     idx = 0
#     W1 = tf.reshape(params[idx: idx + D * hidden_dim], (D, hidden_dim))
#     idx += D * hidden_dim
#     b1 = tf.reshape(params[idx: idx + hidden_dim], (hidden_dim,))
#     idx += hidden_dim
#     W2 = tf.reshape(params[idx: idx + hidden_dim * num_classes], (hidden_dim, num_classes))
#     idx += hidden_dim * num_classes
#     b2 = tf.reshape(params[idx: idx + num_classes], (num_classes,))
    
#     # 순전파 계산
#     hidden = tf.nn.relu(tf.matmul(X_batch, W1) + b1)
#     logits = tf.matmul(hidden, W2) + b2
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_batch, logits=logits))
#     return loss

# def compute_validation_loss(params):
#     """
#     최종 파라미터를 사용해 validation (테스트) 데이터에 대한 loss를 계산합니다.
#     """
#     idx = 0
#     W1 = tf.reshape(params[idx: idx + D * hidden_dim], (D, hidden_dim))
#     idx += D * hidden_dim
#     b1 = tf.reshape(params[idx: idx + hidden_dim], (hidden_dim,))
#     idx += hidden_dim
#     W2 = tf.reshape(params[idx: idx + hidden_dim * num_classes], (hidden_dim, num_classes))
#     idx += hidden_dim * num_classes
#     b2 = tf.reshape(params[idx: idx + num_classes], (num_classes,))
    
#     hidden = tf.nn.relu(tf.matmul(X_test_tf, W1) + b1)
#     logits = tf.matmul(hidden, W2) + b2
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_test_tf, logits=logits))
#     return loss

# def value_and_gradients_function(params):
#     """
#     미니배치를 사용하여 loss와 gradient를 함께 계산합니다.
#     tfp.optimizer.lbfgs_minimize에서 요구하는 형태입니다.
#     """
#     with tf.GradientTape() as tape:
#         tape.watch(params)
#         loss = neural_net_loss_minibatch(params)
#     grad = tape.gradient(loss, params)
#     return loss, grad

# # 학습 시작 시간 측정
# start_time = time.time()

# # L-BFGS 최적화 수행 (미니배치)
# results = tfp.optimizer.lbfgs_minimize(
#     value_and_gradients_function=value_and_gradients_function,
#     initial_position=tf.convert_to_tensor(init_params, dtype=tf.float32),
#     max_iterations=100
# )

# # 학습 종료 시간 측정
# end_time = time.time()
# training_time = end_time - start_time

# # 최적화 결과로 최종 파라미터 추출
# opt_params = results.position

# # 최종 미니배치 기반 training loss와 validation loss 계산
# final_train_loss = neural_net_loss_minibatch(opt_params).numpy()
# final_val_loss = compute_validation_loss(opt_params).numpy()

# print("Final Training Loss (mini-batch): {:.4f}".format(final_train_loss))
# print("Final Validation Loss: {:.4f}".format(final_val_loss))
# print("Total Iterations: {}".format(results.num_iterations))
# print("Training time: {:.2f} seconds".format(training_time))