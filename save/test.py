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

def compute_tensor_W(W):
    """
    W: (m, p) 행렬
    반환: (m, p, p) 텐서 T, 여기서 T[a-1] = W_a^T * W_a
    """
    # 각 행에 대해 외적(outer product) 계산
    # np.einsum을 사용하면 효율적으로 계산할 수 있습니다.
    T = np.einsum('ij,ik->ijk', W, W)
    return T

def compute_tensor(W, y):
    """
    주어진 m x p 행렬 W와 p x l 행렬 y에 대해,
    Tensor_{ak} = sum_{p,j} (W_{ap} * W_{aj} * y_{jk} * (δ_{jp} - y_{pk}))
    를 계산하여 m x l 행렬로 반환한다.
    
    파라미터:
      W : numpy.ndarray of shape (m, p)
      y : numpy.ndarray of shape (p, l)
    
    반환:
      tensor : numpy.ndarray of shape (m, l)
    """
    # 첫 번째 항: 각 (a,k)에 대해 ∑_p W[a,p]² * y[p,k]
    term1 = np.dot(W**2, y)
    
    # 두 번째 항: 각 (a,k)에 대해 (∑_p W[a,p]*y[p,k])²
    term2 = (np.dot(W, y))**2
    
    return term1 - term2

# 예시 사용:
if __name__ == '__main__':
    # 예시 행렬 크기 설정: W는 m x p, y는 p x l
    m, p, l = 3, 4, 2
    
    # 임의의 값을 갖는 행렬 생성
    W = np.random.rand(m, p)
    y = np.random.rand(p, l)
    
    tensor = compute_tensor(W, y)
    
    print("W ({} x {}):".format(m, p))
    print(W)
    print("\ny ({} x {}):".format(p, l))
    print(y)
    print("\nComputed Tensor ({} x {}):".format(m, l))
    print(tensor)