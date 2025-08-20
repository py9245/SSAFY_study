import math
# for i in range(1, 4):
#     for j in range(1, 4):
#         if j != i:
#             for k in range(1, 4):
#                 if k != j and k != i:
#                     print(i, j, k)


def perm(selected, remain, cnt):
    if not remain:
        print(selected)
    else:
        print(f"{'----' * cnt} selected: {selected}, remain: {remain}")
        for i in range(len(remain)):
            select_i = remain[i]
            remain_list = remain[:i] + remain[i + 1:]
            perm(selected + [select_i], remain_list, cnt + 1)


perm([], [1, 2, 3, 4], 1)
print(math.factorial(4))