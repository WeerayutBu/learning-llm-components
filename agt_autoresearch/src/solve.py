# EDITABLE — autoresearch rewrites this in place. Karpathy's train.py.


def solve(n):
    return sum(i for i in range(n) if i % 3 == 0 or i % 5 == 0)
