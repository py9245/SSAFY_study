class Stack:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.items = [None] * capacity
        self.top = -1

    def is_full(self):
        return self.top == self.capacity - 1

    def push(self, item):
        if self.is_full():
            raise IndexError('Stack is full')
        self.top += 1
        self.items[self.top] = item

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        item = self.items[self.top]
        self.items[self.top] = None
        self.top -= 1

    def is_empty(self):
        return self.top == -1

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[self.top]


string = '(6+5*(2-8)/2)'
ls = len(string)

oper = Stack(ls)
after_nums = Stack(ls)

oper_dict = {'(': 0, "*": 2, "/": 2, "+": 1, "-": 1}

for s in string:
    if s.isdigit():
        after_nums.push(s)
    else:
        if oper.is_empty() or s == '(': # oper 비어있으면
            oper.push(s) #
        elif s == ')':
            o = oper.pop()
            while o != '(':
                after_nums.push(o)
                o = oper.pop()
        else:
            ss = oper_dict[s]
            o = oper.pop()
            print(o)
            while oper_dict[o] >= ss:
                after_nums.push(o)
                o = oper.pop()
                if o in ')(':
                    break
            oper.push(o)
print(after_nums.items)