# how to caculate integration
# eg. x^2+ sin x
import numpy as np


def quad(f, a, b, n):  # function of integrate
    h = (b - a) / n
    sum = 0
    for ii in range(0, n):
        sum = sum + f(a + (ii + 0.5) * h)
    return sum * h


fun = lambda x: x**2 + np.sin(x)
res = quad(fun, -2, 2, 20000)
print("Integral is: ", res)
print("error:", 16 / 3 - res)
# def trapezoidal()
