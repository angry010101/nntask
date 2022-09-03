import numpy as np


def checkIsland(map, i, j, n, m):
    if map[i][j] == 1:
        map[i][j] = 0
        if i + 1 < n:
            checkIsland(map, i + 1, j, n, m)
        if i - 1 >= 0:
            checkIsland(map, i - 1, j, n, m)
        if j + 1 < m:
            checkIsland(map, i, j + 1, n, m)
        if j - 1 >= 0:
            checkIsland(map, i, j - 1, n, m)
        return 1
    else:
        return 0


def count(map, n, m):
    res = 0
    for i in range(0, n):
        for j in range(0, m):
            res += checkIsland(map, i, j, n, m)
    return res


def program():
    print("Input")
    n = int(input("Enter N:"))
    m = int(input("Enter M:"))
    map = np.zeros((n, m))
    for i in range(0, n):
        for j in range(0, m):
            map[i][j] = int(input(""))
    print(f"Islands: {count(map, n, m)}")


if __name__ == '__main__':
    program()
