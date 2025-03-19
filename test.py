import sys

import matplotlib.pyplot as plt
import numpy as np

# 增加递归深度限制
sys.setrecursionlimit(3000)


def ACK(x, y):
    stack = [(x, y)]
    while stack:
        x, y = stack.pop()
        if x == 0:
            result = y + 1
        elif y == 0:
            stack.append((x - 1, 1))
            continue
        else:
            stack.append((x - 1, None))
            stack.append((x, y - 1))
            continue

        while stack and stack[-1][1] is None:
            x, _ = stack.pop()
            result = ACK(x, result)

        if stack:
            stack[-1] = (stack[-1][0], result)

    return result


if __name__ == "__main__":
    n = 4
    ack = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            ack[i][j] = ACK(i, j)
    print(ack)
    plt.imshow(ack, cmap='hot', interpolation='nearest')
    plt.show()
