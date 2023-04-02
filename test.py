from memory_profiler import profile

@profile
def test():
    print('aga')
    a=2*2
    b=a*2

test()