import numpy as np

# 기본 생성
# print(np.array([1, 2, 3]))             # 리스트에서 배열 생성
# print(np.zeros((3, 4)))       # 0으로 채운 배열
# print(np.ones((2, 3)))                  # 1로 채운 배열
# print(np.full((2, 2), 7))               # 특정 값으로 채운 배열
# print(np.eye(3))                        # 단위 행렬
# print(np.arange(0, 10, 2))              # [0, 2, 4, 6, 8]
# print(np.linspace(0, 1, 5))

# 랜덤 생성 (AI/ML에서 매우 중요!)
a = np.random.rand(3, 4)             # 0~1 균등분포
b = np.random.randn(3, 4)            # 표준정규분포
c = np.random.randint(0, 10, (3, 4)) # 정수 랜덤
d = np.random.choice([1,2,3], 5)     # 랜덤 샘플링
e = np.random.seed(42)               #

print(f"a : {a}")
print(f"b : {b}")
print(f"c : {c}")
print(f"d : {d}")
print(f"e : {e}")