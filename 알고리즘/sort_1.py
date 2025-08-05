# 버블 정렬
def buble(arr):
    n = len(arr)
    for i in range(n):
        cnt = 0
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                cnt += 1
        if not cnt:
            break
    return arr
arr = [55, 7, 78, 12, 42]
print(buble(arr))