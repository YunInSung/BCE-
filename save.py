import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time

# 상수 및 데이터셋 설정
N = 5000         # 데이터 샘플 수
D = 40            # 입력 차원
hidden_dim = 80
iterator = 25
num_classes = 20  # 클래스 수
epsilon = 1e-16
N_float = tf.cast(N, tf.float32)

# 데이터 생성 (NumPy)
np.random.seed(129)
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
def grad_W2_func_r_tf_batch(h, row, d2Z2, A1i):
    """
    h:     공통 입력 텐서, shape = (80, 5000)
    d2Z2:  전체 d2Z2 텐서, shape = (p, 5000)
    A1i:   보조 입력 텐서, shape = (81, 5000)
    row:   배치 인덱스, shape = (B,)
    N_float: 스칼라 상수

    출력:
      각 배치에 대해 계산한 결과, shape = (B, 81, 80)
      (원래 단일 케이스에서는 결과가 (81,80)로 나오므로)
    """
    # d2Z2_batch: (B, 5000)
    d2Z2_batch = tf.gather(d2Z2, row)
    
    # 우리가 원하는 것은, 각 배치 i에 대해 아래 연산을 수행하는 것:
    # new_h = (d2Z2[i] * h) / N_float    # h: (80,5000), d2Z2[i]: (5000,) → new_h: (80,5000)
    # result = tf.expand_dims(new_h, axis=1) * tf.expand_dims(A1i, axis=0)
    #           → (80,1,5000) * (1,81,5000) = (80,81,5000)
    # res_sum = tf.reduce_sum(result, axis=2) → (80,81)
    # final = tf.transpose(res_sum) → (81,80)
    
    def process_single(i):
        # d2Z2_i: (5000,)
        d2Z2_i = d2Z2_batch[i]
        # new_h: (80,5000)
        new_h = (d2Z2_i * h) / N_float
        # expand: new_h → (80,1,5000), A1i → (1,81,5000)
        res = tf.expand_dims(new_h, axis=1) * tf.expand_dims(A1i, axis=0)
        # res: (80,81,5000) → reduce sum over axis=2 → (80,81)
        res_sum = tf.reduce_sum(res, axis=2)
        # transpose: (81,80)
        return tf.transpose(res_sum)
    
    # new_h_batch will have shape (B, 81, 80)
    matrix = tf.map_fn(process_single, tf.range(tf.shape(d2Z2_batch)[0]), dtype=tf.float32)
    return matrix


def H_matirx2_r_tf_batch(h, row, d2Z2, A1i, grad_W2_r):
    """
    입력:
      h:       보조 입력 텐서 (원래 h의 shape에 따라 m을 유추), 예: (M,) 또는 (1, M)
      row:     배치 인덱스, shape (B,)
      d2Z2:    shape (p, M)
      A1i:     shape (m+1, M)
      grad_W2_r: 배치별 grad_W2 결과, 예상 shape: (B, m+1, m)
    출력:
      배치별 H 행렬, 예상 shape: (B, m+1, m+1)
    """
    B = tf.shape(row)[0]
    # out_dim: m+1 (출력 차원)
    out_dim = tf.shape(A1i)[0]
    # d2Z2_batch: (B, M)
    d2Z2_batch = tf.gather(d2Z2, row)
    # A1i: (m+1, M) → (1, m+1, M)
    A1i_exp = tf.expand_dims(A1i, 0)
    # d2Z2_batch: (B, M) → (B, 1, M)
    d2Z2_exp = tf.expand_dims(d2Z2_batch, 1)
    # 곱: (B, m+1, M)
    prod = d2Z2_exp * A1i_exp
    # M 축 합: (B, m+1)
    summed = tf.reduce_sum(prod, axis=2)
    # reshape: (B, m+1, 1)
    sum_tensor = tf.reshape(summed / N_float, [B, out_dim, 1])
    # grad_W2_r: (B, m+1, m)와 sum_tensor: (B, m+1, 1)를 concat (axis=2) → (B, m+1, m+1)
    matrix = tf.concat([grad_W2_r, sum_tensor], axis=2)
    return matrix


def L_matrix2_r_tf_batch(h, P, row, dW2, db2, d2Z2, A1i, grad_W2_r, learn):
    """
    입력:
      h:       보조 입력 텐서, (M,) 또는 (1, M)
      P:       파라미터 텐서, shape (p, m+1)
      row:     배치 인덱스, shape (B,)
      dW2:     변화량 텐서, shape (p, m) (또는 이에 상응하는 shape)
      db2:     변화량 텐서, shape (p, 1)
      d2Z2:    텐서, shape (p, M)
      A1i:     텐서, shape (m+1, M)
      grad_W2_r: 배치별 grad_W2, 예상 shape: (B, m+1, m)
      learn:   스칼라 학습률
    출력:
      배치별 L 행렬, 예상 shape: (B, 1, m+1)
    """
    B = tf.shape(row)[0]
    out_dim = tf.shape(A1i)[0]  # m+1
    # 배치별로 v0, W2_r, db2 추출: 
    v0 = tf.gather(P, row)      # (B, m+1)
    W2_r = tf.gather(dW2, row) * learn  # (B, m)  (dW2의 원래 shape에 맞게)
    
    # v0를 (B, 1, m+1)로 확장
    v0_exp = tf.expand_dims(v0, axis=1)  # (B, 1, m+1)
    # grad_W2_r: (B, m+1, m)
    mat = tf.matmul(v0_exp, grad_W2_r)  # (B, 1, m)
    mat_flat = tf.reshape(mat, [B, -1])  # (B, m)
    matrix_part = mat_flat - W2_r        # (B, m)
    
    # grad_b_r: 배치별로 계산. 먼저 d2Z2와 A1i의 배치별 곱셈
    d2Z2_batch = tf.gather(d2Z2, row)   # (B, M)
    # A1i: (m+1, M) → (1, m+1, M)
    A1i_exp = tf.expand_dims(A1i, 0)
    # d2Z2_batch: (B, M) → (B, 1, M)
    d2Z2_exp = tf.expand_dims(d2Z2_batch, 1)
    prod = d2Z2_exp * A1i_exp           # (B, m+1, M)
    summed = tf.reduce_sum(prod, axis=2)  # (B, m+1)
    grad_b_r = tf.reshape(summed / N_float, [B, out_dim, 1])  # (B, m+1, 1)
    
    # b_r: (B, 1)
    b_r = tf.gather(db2, row) * learn    # (B, 1)
    
    # v0_exp: (B, 1, m+1)와 grad_b_r: (B, m+1, 1)의 matmul → (B, 1, 1)
    tmp = tf.matmul(v0_exp, grad_b_r)    # (B, 1, 1)
    tmp = tf.reshape(tmp, [B, 1]) - b_r   # (B, 1)
    
    # matrix_part: (B, m)와 tmp: (B, 1)을 concat하여 (B, m+1)
    combined = tf.concat([matrix_part, tmp], axis=1)  # (B, m+1)
    # 확장: (B, 1, m+1)
    combined = tf.expand_dims(combined, axis=1)
    return combined


##############################################
#       TensorFlow 버전의 W1, b1 관련 함수      #
##############################################
def grad_W1_func_r_tf(theta, X, Xi):
    """
    입력:
      theta: (B, M)
      X:     (D, M)
      Xi:    (D+1, M)
    출력:
      각 배치에 대해 계산한 결과, shape: (B, D+1, D)
    """
    # X와 Xi를 배치 차원에 맞게 확장 (자동 브로드캐스트 활용)
    X_b = tf.expand_dims(X, 0)      # (1, D, M) -> broadcast to (B, D, M)
    Xi_b = tf.expand_dims(Xi, 0)    # (1, D+1, M) -> broadcast to (B, D+1, M)

    # theta: (B, M) -> (B, 1, M)로 확장
    theta_exp = tf.expand_dims(theta, 1)  # (B, 1, M)
    
    # tmp: 각 배치별로 X와 곱함. 결과: (B, D, M)
    tmp = theta_exp * X_b / N_float

    # 원래 코드에서는 tmp에 대해 expand_dims 후 Xi와 elementwise 곱해 reduce_sum 하였음.
    # tmp: (B, D, M) -> (B, D, 1, M)
    tmp_exp = tf.expand_dims(tmp, 2)  # (B, D, 1, M)
    # Xi_b: (B, D+1, M) -> (B, 1, D+1, M)
    Xi_exp = tf.expand_dims(Xi_b, 1)  # (B, 1, D+1, M)
    
    # elementwise 곱: (B, D, D+1, M)
    result = tmp_exp * Xi_exp
    # M 축을 따라 합산: (B, D, D+1)
    result_sum = tf.reduce_sum(result, axis=-1)
    # 각 배치별로 transpose (D, D+1) -> (D+1, D)
    matrix = tf.transpose(result_sum, perm=[0, 2, 1])
    return matrix  # (B, D+1, D)


def H_matirx1_r_tf(theta, X, Xi, grad_W1_r):
    """
    입력:
      theta:     (B, M)
      X:         (D, M)    -- 단일 입력, 공통 사용
      Xi:        (D+1, M)  -- 단일 입력, 공통 사용
      grad_W1_r: (B, D+1, D) from grad_W1_func_r_tf
    출력:
      각 배치별로 구성된 행렬, shape: (B, D+1, D+1)
    """
    # theta: (B, M) -> (B, 1, M)
    theta_exp = tf.expand_dims(theta, 1)
    # Xi: (D+1, M) -> (1, D+1, M)
    Xi_exp = tf.expand_dims(Xi, 0)
    # 배치마다 곱: (B, D+1, M)
    prod = theta_exp * Xi_exp
    # M 축 합산 -> (B, D+1)
    sum_tensor = tf.reduce_sum(prod, axis=-1)
    # N_float로 나누고 (B, D+1, 1)로 reshape
    sum_tensor = tf.reshape(sum_tensor / N_float, [tf.shape(theta)[0], -1, 1])
    
    # grad_W1_r: (B, D+1, D)와 sum_tensor: (B, D+1, 1)을 concat하면 (B, D+1, D+1)
    matrix = tf.concat([grad_W1_r, sum_tensor], axis=2)
    return matrix  # (B, D+1, D+1)


def L_matrix1_r_tf(X, P, theta, dW1, db1, row, Xi, grad_W1_r, learn):
    """
    입력:
      X:         (D, M)         -- 공통 입력
      P:         (m, D+1)       -- 각 행에 대한 파라미터 집합
      theta:     (B, M)         -- 배치별 theta
      dW1:       (m, D)         -- 각 행에 대한 변화량
      db1:       (m, 1)         -- 각 행에 대한 변화량
      row:       (B,)           -- 배치에 해당하는 인덱스들
      Xi:        (D+1, M)       -- 공통 입력
      grad_W1_r: (B, D+1, D)    -- 배치별 grad_W1 (위 함수 결과)
      learn:     스칼라 학습률
    출력:
      각 배치별 L 벡터, shape: (B, 1, D+1)
    """
    B = tf.shape(row)[0]
    D = tf.shape(X)[0]
    
    # P, dW1, db1에서 배치에 해당하는 행들 추출
    v0 = tf.gather(P, row)      # (B, D+1)
    W1_r = tf.gather(dW1, row) * learn  # (B, D) -- dW1의 shape에 따라 다름
    # v0를 (B, 1, D+1)로 확장하여 배치별 matmul 수행
    v0_exp = tf.expand_dims(v0, axis=1)  # (B, 1, D+1)
    # grad_W1_r: (B, D+1, D)
    mat = tf.matmul(v0_exp, grad_W1_r)  # (B, 1, D)
    # 결과를 (B, D)로 평탄화
    mat_flat = tf.reshape(mat, [B, -1])  # (B, D)
    # matrix_part = mat_flat - W1_r, (B, D)
    matrix_part = mat_flat - W1_r

    # grad_b_r: 계산은 theta와 Xi를 이용 (B, M)와 (D+1, M)
    theta_exp = tf.expand_dims(theta, 1)  # (B, 1, M)
    Xi_exp = tf.expand_dims(Xi, 0)          # (1, D+1, M)
    prod = theta_exp * Xi_exp              # (B, D+1, M)
    grad_b_r = tf.reduce_sum(prod, axis=-1) / N_float  # (B, D+1)
    grad_b_r = tf.expand_dims(grad_b_r, axis=2)        # (B, D+1, 1)
    
    # b_r: (B, 1)
    b_r = tf.gather(db1, row) * learn  # (B, 1)
    
    # 배치별로 v0와 grad_b_r를 matmul: v0_exp: (B, 1, D+1)와 grad_b_r: (B, D+1, 1)
    tmp = tf.matmul(v0_exp, grad_b_r)  # (B, 1, 1)
    tmp = tf.reshape(tmp, [B, 1]) - b_r  # (B, 1)
    
    # matrix_part: (B, D)와 tmp: (B, 1)을 concat하여 (B, D+1)
    combined = tf.concat([matrix_part, tmp], axis=1)  # (B, D+1)
    # 확장해서 최종 출력: (B, 1, D+1)
    combined = tf.expand_dims(combined, axis=1)
    return combined


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

    # hidden row 처리를 위한 함수 (기존 process_hidden_row와 동일)
    def process_hidden_row_batch(row_idxs):
        # tf.gather로 각 배치에 해당하는 값들 추출 (예: J_1, df1, J_2, d2f1)
        batch_J_1  = tf.gather(J_1, row_idxs)   # (B, M)
        batch_df1  = tf.gather(df1, row_idxs)   # (B, M)
        batch_J_2  = tf.gather(J_2, row_idxs)   # (B, M)
        batch_d2f1 = tf.gather(d2f1, row_idxs)  # (B, M)
        
        # theta 계산: (B, M)
        theta = (batch_J_1 * (batch_df1 ** 2)) + (batch_J_2 * batch_d2f1)
        
        # 위에서 정의한 배치 연산 함수들 호출
        grad_W1_r = grad_W1_func_r_tf(theta, X, Xi)        # (B, D+1, D)
        H_r       = H_matirx1_r_tf(theta, X, Xi, grad_W1_r)  # (B, D+1, D+1)
        L_r       = L_matrix1_r_tf(X, P1, theta, dW1, db1, row_idxs, Xi, grad_W1_r, learn)  # (B, 1, D+1)
        
        # 예: H_r_reg 계산 (각 배치마다 단위 행렬 더하기)
        def add_reg(h):
            return h + epsilon * tf.eye(tf.shape(h)[0], dtype=tf.float32)
        H_r_reg = tf.map_fn(add_reg, H_r)  # (B, D+1, D+1)
        
        # solve 연산: 배치별로 진행
        def solve_single(inputs):
            h_reg, l = inputs  # h_reg: (D+1, D+1), l: (1, D+1)
            sol = tf.linalg.solve(tf.transpose(h_reg), tf.transpose(l))
            return tf.transpose(sol)  # (1, D+1)
        
        P_r = tf.map_fn(solve_single, (H_r_reg, L_r), dtype=tf.float32)  # (B, 1, D+1)
        return P_r


    hidden_rows = tf.range(m)  # m = tf.shape(W1)[0]
    matrix1 = process_hidden_row_batch(hidden_rows)
    # matrix1 = tf.vectorized_map(lambda idx: process_hidden_row_batch(idx), hidden_rows)
    matrix1 = tf.squeeze(matrix1, axis=1)  # (m, D+1)
    # 결과 shape: (m, 1, D+1) → squeeze
    cpW1 = matrix1[:, :n]         # (m, D)
    cpb1 = tf.reshape(matrix1[:, n], [m, 1])  # (m, 1)

    i_A1 = tf.ones((1, tf.shape(A1)[1]), dtype=tf.float32)
    A1i = tf.concat([A1, i_A1], axis=0)  # (m+1, N)
    P2 = tf.concat([W2, b2], axis=1)      # (p, m+1)
    # 출력층 업데이트: 각 output row별 계산을 tf.map_fn으로 처리
    def process_output_row_batch(row_idxs):
        """
        row_idxs: 배치 인덱스, shape (B,)
        아래 함수들은 위에서 정의한 배치 버전을 사용
        """
        # grad_W2_r: (B, m+1, m)
        grad_W2_r_batch = grad_W2_func_r_tf_batch(A1, row_idxs, d2Z2, A1i)
        # H 행렬: (B, m+1, m+1)
        H_r_batch = H_matirx2_r_tf_batch(A1, row_idxs, d2Z2, A1i, grad_W2_r_batch)
        # L 행렬: (B, 1, m+1)
        L_r_batch = L_matrix2_r_tf_batch(A1, P2, row_idxs, dW2, db2, d2Z2, A1i, grad_W2_r_batch, learn)
        
        # 각 배치에 대해 정규화된 H 행렬: H_r_reg = H_r + epsilon * I
        def add_reg(h):
            return h + epsilon * tf.eye(tf.shape(h)[0], dtype=tf.float32)
        H_r_reg_batch = tf.map_fn(add_reg, H_r_batch)  # (B, m+1, m+1)
        
        # 각 배치별로 solve 연산 수행
        def solve_single(inputs):
            h_reg, l = inputs  # h_reg: (m+1, m+1), l: (1, m+1)
            sol = tf.linalg.solve(tf.transpose(h_reg), tf.transpose(l))
            return tf.transpose(sol)  # (1, m+1)
        
        P_r_batch = tf.map_fn(solve_single, (H_r_reg_batch, L_r_batch), dtype=tf.float32)
        return P_r_batch  # (B, 1, m+1)

    output_rows = tf.range(p)  # p: 전체 출력 행 수 (예: tf.shape(Y)[0])
    matrix2 = process_output_row_batch(output_rows)
    # matrix2 = tf.vectorized_map(lambda idx: process_output_row(idx), output_rows)
    matrix2 = tf.squeeze(matrix2, axis=1)  # (p, m+1)
    cpW2 = matrix2[:, :m]         # (p, m)
    cpb2 = tf.reshape(matrix2[:, m], [p, 1])
    return cpW1, cpb1, cpW2, cpb2


def ret_weight_tf(X, Y, W1, b1, W2, b2, loss0, iter=1):
    prev_loss = loss0
    loss = tf.constant(1.0, dtype=tf.float32)
    cpW1 = tf.identity(W1)
    cpb1 = tf.identity(b1)
    cpW2 = tf.identity(W2)
    cpb2 = tf.identity(b2)
    prevW1, prevb1, prevW2, prevb2 = cpW1, cpb1, cpW2, cpb2
    learn = 0.005
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
        if continuous >= 2 and learn < 0.35:
            learn *= 2
            if learn > 0.35 :
              learn = 0.35
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
lr = 0.1
epochs = 1200
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
