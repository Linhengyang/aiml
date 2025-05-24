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

    y = compile(prog, '', 'exec')

    exec(y)

