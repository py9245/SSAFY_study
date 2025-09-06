# def comb(arr, n):
#     result = []  # 조합을 저장할 리스트
#
#     if n == 1:
#         return [[i] for i in arr]
#
#     for i in range(len(arr)):
#         elem = arr[i]
#         print(arr[i + 1:], n - 1)
#         for rest in comb(arr[i + 1:], n - 1):  # 조합
#             # for rest in comb(arr[:i] + arr[i+1:], n - 1):  # 순열
#             # for rest in comb(arr, n - 1):  # 중복순열
#             # for rest in comb(arr[i:], n - 1):  # 중복조합
#             result.append([elem] + rest)
#
#     return result
#
#
# print(comb([1, 2, 3, 4], 3))


arr = [1,2,3,4,5,6,7,8,9,10]
result = []

for i in range(1<<len(arr)):
    add_list = []
    for j in range(len(arr)):
        if i & (1 << j):
            add_list.append(arr[j])
    if sum(add_list) == 10:
        result.append(add_list)

print(result)