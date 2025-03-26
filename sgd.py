import numpy as np
import matplotlib.pyplot as plt

# 최적화할 함수와 기울기 정의
def f(x):
    return (x - 3) ** 2

def grad_f(x):
    return 2 * (x - 3)

# SGD 구현
def sgd(initial_x, learning_rate, epochs):
    x = initial_x
    history = [x]
    
    for epoch in range(epochs):
        # 각 반복마다 현재 위치의 기울기를 계산 후 파라미터 업데이트
        grad = grad_f(x)
        x = x - learning_rate * grad
        history.append(x)
    
    return x, history

# 파라미터 설정
initial_x = 0.0   # 초기값
learning_rate = 0.1
epochs = 30

# SGD 실행
optimal_x, history = sgd(initial_x, learning_rate, epochs)

print("최적의 x 값:", optimal_x)
print("함수 값:", f(optimal_x))

# SGD 경로 시각화
x_values = np.linspace(-1, 7, 400)
y_values = f(x_values)
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label='f(x) = (x-3)^2')
plt.scatter(history, [f(x) for x in history], color='red', zorder=5, label='SGD 경로')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('SGD를 이용한 최적화 경로')
plt.legend()
plt.show()
