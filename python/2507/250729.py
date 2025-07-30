a = 1 # 1
b = 2 # 2

def en(): # 4
    a = 10 # 5
    c = 3 # 6

    def lo(c): # 8
        print(a, b, c) # 9
    lo(500) # 7
    print(a, b, c) # 10
en() # 3
print(a, b) # 11