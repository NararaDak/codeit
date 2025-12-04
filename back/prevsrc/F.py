import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lib.QS as qs
titanic = pd.read_csv('D:/01.project/코드잇/part-1-main/data/titanic.csv')
print(titanic.head())
print(titanic.tail())
print(titanic.shape)
qs.Lines()
print(titanic.columns)
qs.Lines()
print(titanic[titanic['Age'].notna()])

#################################################
arr = plt.imread('D:/01.project/코드잇/part-1-main/image/dog_gray.jpg')
arr = arr.copy()  # 배열을 복사하여 쓰기 가능하게 만듦
arr = arr//2


print(arr.shape)
h, w = arr.shape[:2]  # 이미지의 높이와 너비
center_h, center_w = h // 2, w // 2

arr[center_h-10:center_h+10, center_w-10:center_w+10] = 0

plt.imshow(arr, cmap='gray')
# plt.show()
#################################################
N = 1_000_000

# 난수 생성: 0과 1 사이의 실수 N개를 각각 x, y에 저장
x = np.random.rand(N)
y = np.random.rand(N)

# 각 점(x, y)이 원의 1/4 내부에 있는지 여부를 계산
# (x^2 + y^2 <= 1)이면 True, 아니면 False
inside = (x**2 + y**2) <= 1.0
print(inside)  # True/False로 이루어진 배열 출력

# 원 내부에 들어간 점의 개수를 전체 점의 개수로 나누고 4를 곱해 π(파이)를 추정
mypi = (inside.sum() / N) * 4

print("추정한 원주율:", mypi)  # 추정된 π 값 출력

#################################################

# 1부터 1000까지의 정수로 이루어진 배열 생성
arr = np.arange(1, 1001)

# arr을 행(row)으로 변환: (1, 1000) 모양
row = arr[None, :]  # 2차원 배열로 만들어서 한 행에 모든 값이 들어감

# arr을 열(col)으로 변환: (1000, 1) 모양
col = arr[:, None]  # 2차원 배열로 만들어서 한 열에 모든 값이 들어감

# 곱셈표(구구단처럼) 만들기: 각 행의 값과 각 열의 값을 곱함
table = row * col  # 브로드캐스팅을 이용해 (1000, 1000) 행렬 생성

# 각 배열의 모양(크기) 출력
print(row.shape, col.shape, table.shape)  # (1, 1000), (1000, 1), (1000, 1000)

# 곱셈표 전체 출력
print(table)

print(table.info())

