def max_iter(m=1000):
    def f(k, *args, **kwargs):
        return k > m

    return f


