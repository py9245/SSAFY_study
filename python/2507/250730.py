# def sol(age):
#     print(f"age : {age}")
    
# print(sol(24))

# def great(name = "민수", age = 20):
#     print(name, age)
    
# great(20)

# print("123", 1)

# A = {1,2,3,4,5}
# print()
# a, *b, c = A
# print(a,b,c)

# A = [1,2,3]
# B = A
# C = [1,2,3]
# print(id(A))
# print(id(B))
# print(id(C))
# print()
# print(id(A[0]))
# print(id(B[0]))
# print(id(C[0]))
# print(id(1))
# x = 140725989897000

import copy

a = [1,2,3]
b = copy.deepcopy(a)
print(id(a))
print(id(b))
for i in range(3):
    print(id(a[i]))
    print(id(b[i]))