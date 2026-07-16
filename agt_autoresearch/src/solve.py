# EDITABLE — autoresearch rewrites this in place. Karpathy's train.py.


def solve(n):
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    i = 2
    while i * i < n:
        if sieve[i]:
            for j in range(i * i, n, i):
                sieve[j] = False
        i += 1
    return sum(sieve)
