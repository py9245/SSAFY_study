def infix_to_postfix(expression):
    op_dict = {'(': 0, "*": 2, "/": 2, "+": 1, "-": 1}
    postfix = []
    stack = []

    for char in expression:
        if char.isnumeric():
            postfix.append(char)
        elif char == '(':
            stack.append(char)
        elif char == ")":
            pop_token = stack.pop()
            while pop_token != "(":
                postfix.append(pop_token)
                pop_token = stack.pop()
        else:
            while stack and op_dict[stack[-1]] >= op_dict[char]:
                postfix.append(stack.pop())
            stack.append(char)
    while stack:
        postfix.append(stack.pop())

    return ''.join(postfix)


def run_calculator(expr):
    stack = []
    tokens = expr.split()

    for token in tokens:
        if token.isnumeric():
            stack.append(int(token))
        else:
            op2 = stack.pop()
            op1 = stack.pop()

            if token == "*":
                stack.append(op1 * op2)
            elif token == "/":
                stack.append(op1 // op2)
            elif token == "+":
                stack.append(op1 + op2)
            else:
                stack.append(op1 - op2)
    return stack.pop()


string = "(6+5*(2-8)/2)"
strinf_fix = infix_to_postfix(string)
print(strinf_fix)
refix = run_calculator(strinf_fix)
print(refix)

