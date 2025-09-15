# 선택 정렬
def choies(arr):
    n = len(arr)
    for i in range(1, n):
        for j in range(i, 0, -1):
            if arr[j - 1] > arr[j]:
                arr[j - 1], arr[j] = arr[j], arr[j - 1]
            else :
                pass
    return arr

arr = [535, 17, 1000,  788, 122, 42, 0]
print(choies(arr))