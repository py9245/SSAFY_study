import math
# for i in range(1, 4):
#     for j in range(1, 4):
#         if j != i:
#             for k in range(1, 4):
#                 if k != j and k != i:
#                     print(i, j, k)


# def perm(selected, remain, cnt):
#     if not remain:
#         print(selected)
#     else:
#         print(f"{'----' * cnt} selected: {selected}, remain: {remain}")
#         for i in range(len(remain)):
#             select_i = remain[i]
#             remain_list = remain[:i] + remain[i + 1:]
#             perm(selected + [select_i], remain_list, cnt + 1)
#
#
# perm([], [1, 2, 3, 4], 1)
# print(math.factorial(4))

def comb(arr, n):
    result = []  # 조합을 저장할 리스트

    if n == 1:
        return [[i] for i in arr]

    for i in range(len(arr)):
        elem = arr[i]

        for rest in comb(arr[i + 1:], n - 1):  # 조합
            # for rest in comb(arr[:i] + arr[i+1:], n - 1):  # 순열
            # for rest in comb(arr, n - 1):  # 중복순열
            # for rest in comb(arr[i:], n - 1):  # 중복조합
            result.append([elem] + rest)

    return result

for i in range(5):
    print(comb([1, 2, 3, 4], i))
