# 선택 정렬
def count(arr):
    n = len(arr)
    max_val = max(arr)
    cnt = [0] * (max_val + 1)
    for num in arr:
        cnt[num] += 1
    
    for c in range(1, max_val + 1):
        cnt[c] += cnt[c - 1]

    new_arr = [0] * n
    for a in range(n - 1, -1, -1):
        var = arr[a] 
        cnt[var] -= 1
        new_arr[cnt[var]] = var
    return new_arr

arr = [5, 7, 10, 8, 2, 2, 0, 1, 1, 1, 1, 2, 3, 6, 4]
print(count(arr))