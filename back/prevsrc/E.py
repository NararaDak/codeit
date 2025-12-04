import numpy as np
import lib.QS as qs
import matplotlib.pyplot as plt


# arr = plt.imread('D:/01.project/코드잇/part-1-main/image/dog_gray.jpg')
# arr = arr.copy()  # 배열을 복사하여 쓰기 가능하게 만듦

# print(arr.shape)

# h, w = arr.shape[:2]  # 이미지의 높이와 너비
# center_h, center_w = h // 2, w // 2

# arr[center_h-10:center_h+10, center_w-10:center_w+10] = 0

# plt.imshow(arr, cmap='gray')
# plt.show()



N = 1_000_000

# 난수 생성
x = np.random.rand(N)
y = np.random.rand(N)
print(x,y)
# 원 내부 조건 (x^2 + y^2 <= 1)
inside = (x**2 + y**2) <= 1

# 비율 계산
pi_est = (inside.sum() / N) * 4

print("추정한 원주율:", pi_est)


arr = np.arange(1,1001)
print(arr)
#print( arr[:,None] )
#print( arr[None,:] )


table = arr[:,None] * arr[None,:]

print(table)