import numpy as np

def compute_tensor(W):
    """
    W: (m, p) 행렬
    반환: (m, p, p) 텐서 T, 여기서 T[a-1] = W_a^T * W_a
    """
    # 각 행에 대해 외적(outer product) 계산
    # np.einsum을 사용하면 효율적으로 계산할 수 있습니다.
    T = np.einsum('ij,ik->ijk', W, W)
    return T

# 예제: m=5, p=3인 행렬 W를 임의로 생성
m, p = 5, 3
W = np.random.rand(m, p)

# 텐서 T 계산
T = compute_tensor(W)

# 예를 들어 첫 번째 행 (1-indexed에서 a=1, 0-indexed에서는 index 0) 의 결과를 확인합니다.
print("W[0]:\n", W[0])
print("\nW[0]^T * W[0]:\n", np.outer(W[0], W[0]))
print("\nT[0]:\n", T[0])


import numpy as np

def compute_J_tensor_vectorized(Y):
    """
    Y: (p, l) 행렬. 각 열이 하나의 y^ (p×1 벡터)를 나타냅니다.
    반환: (p, p, l) 텐서 J, 각 슬라이스 J[:,:,k] = diag(Y[:,k]) - Y[:,k] @ Y[:,k]^T
    """
    p, l = Y.shape
    # 외적 부분: 각 열에 대해 Y[:, k] @ Y[:, k]^T 를 계산합니다.
    outer = np.einsum('ik,jk->ijk', Y, Y)  # shape: (p, p, l)
    
    # 대각 행렬 부분: 각 열에 대해 diag(Y[:, k])를 만듭니다.
    diag_part = np.zeros((p, p, l))
    idx = np.arange(p)
    diag_part[idx, idx, :] = Y  # 각 슬라이스의 대각 성분에 Y의 원소를 배치
    
    J_tensor = diag_part - outer
    return J_tensor

# 사용 예제
p, l = 4, 3  # 예를 들어 p=4, l=3
Y = np.random.rand(p, l)
J_tensor_vectorized = compute_J_tensor_vectorized(Y)
print("J_tensor (벡터화 사용):")
print(J_tensor_vectorized)


import numpy as np

# T: (m, p, p), J: (p, p, l)
# np.tensordot로 1,2번째 축과 0,1번째 축을 합치면,
R = np.tensordot(T, J_tensor_vectorized, axes=([1, 2], [0, 1]))
