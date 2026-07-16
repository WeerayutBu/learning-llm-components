def solve(n):
    n -= 1
    sum3 = (n // 3) * (3 + (n // 3) * 3) // 2
    sum5 = (n // 5) * (5 + (n // 5) * 5) // 2
    sum15 = (n // 15) * (15 + (n // 15) * 15) // 2
    return sum3 + sum5 - sum15