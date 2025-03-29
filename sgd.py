import tensorflow as tf
import numpy as np

# 예시: n x n 행렬 A와 n x 1 벡터 b 생성
n = 5
A = tf.random.normal((n, n), dtype=tf.float32)
b = tf.random.normal((n, 1), dtype=tf.float32)

# tf.linalg.solve를 사용해 A x = b 문제를 풉니다.
x = tf.linalg.solve(A, b)

print("Matrix A:")
print(A.numpy())
print("\nVector b:")
print(b.numpy())
print("\nSolution x (via tf.linalg.solve):")
print(x.numpy())