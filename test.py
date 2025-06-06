# test.py

if __name__ == "__main__":
    def add_():
        return '''
def add(a, b):
    return a + b
'''

    def fancy_func_():
        return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

    def evoke_():
        return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

    prog = evoke_()

    print(prog)
    # from imperative programming to 
    # symbolic programming
    y = compile(prog, '<exec_input>', 'exec') # a code object, executable prorgam, pre-defined before compile

    exec(y)

    # 1. can skip the Python interpretre in many cases: multiple GPUs paired with a single Python thread on a CPU
    # 2. optimize and rewrite code since it can read code entirely
    # 3. easy to port: run the program in a non-Python environment



