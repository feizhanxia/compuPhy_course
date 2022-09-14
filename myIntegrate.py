# how to caculate integration
# eg. x^2+ sin x
import numpy as np
import sys

#  def quad(f, a, b, n):  # function of integrate
#  h = (b - a) / n  # step length
#  sum = 0  # initial integration result = 0
#  for ii in range(0, n):
#  sum = (sum + f(a + (ii + 0.5) * h)) * h
#  return sum

#  def adaptive_quad(f, a, b, eps):
#  it = 0  # initial iteration count = 0
#  maxIt = 100  # maximum iteration times = 100
#  error = sys.float_info.max
#  n = 1
#  vOld = quad(f, a, b, n)
#  while error > eps and maxIt > it:
#  n = n * 2
#  vNew = quad(f, a, b, n)
#  error = np.abs(vNew - vOld)
#  vOld = vNew
#  it += 1
#  print("Iteration:", it)
#  return vOld

#  def simpson(f, a, b, n):
#  h = (b - a) / n
#  sum = 0
#  for ii in range(n):
#  sum += 2 * f(a + ii * h) + 4 * f(a + ii * h + 0.5 * h)
#  sum = h * (sum + f(b) - f(a)) / 6
#  return sum

#  def adaptive_simpson(f, a, b, eps):
#  it = 0  # initial iteration count = 0
#  maxIt = 100  # maximum iteration times = 100
#  error = sys.float_info.max
#  n = 1
#  vOld = simpson(f, a, b, n)
#  while error > eps and maxIt > it:
#  n = n * 2
#  vNew = quad(f, a, b, n)
#  error = np.abs(vNew - vOld)
#  vOld = vNew
#  it += 1
#  print("Iteration:", it)
#  return vOld

# 下面进行面向对象的编程 integration


class myIntegrate():
    def __init__(self, func, a, b):
        self.f = func
        self.a = a
        self.b = b
        self.sum = 0

    def adaptive(self, eps=1e-3, n=1):
        it = 0  # initial iteration count = 0
        maxIt = 100  # maximum iteration times = 100
        error = sys.float_info.max
        vOld = self.caculate(n)
        while error > eps and maxIt > it:
            n = n * 2
            vNew = self.caculate(n)
            error = np.abs(vNew - vOld)
            vOld = vNew
            it += 1
        print("Iteration:", it)
        return vOld


class quad(myIntegrate):
    def __init__(self, func, a, b):
        super().__init__(func, a, b)

    def caculate(self, n):
        h = (self.b - self.a) / n  # step length
        sum = 0  # initial integration result = 0
        for ii in range(n):
            sum = sum + self.f(self.a + (ii + 0.5) * h)
        self.sum = sum * h
        return self.sum


class simpson(myIntegrate):
    def __init__(self, func, a, b):
        super().__init__(func, a, b)

    def caculate(self, n):
        h = (self.b - self.a) / n
        sum = 0
        for ii in range(n):
            sum += 2 * self.f(self.a + ii * h) + 4 * self.f(self.a + ii * h +
                                                            0.5 * h)
        self.sum = h * (sum + self.f(self.b) - self.f(self.a)) / 6
        return self.sum


#  def fun(x):
#  return (np.sin(x)**2)

#  integralFun = simpson(fun, 0, np.pi)
#  res = integralFun.adaptive(1e-5)
#  print("Integral is: ", res)
#  print("error:", np.abs(np.pi / 2 - res))
#  integralFun1 = quad(fun, 0, np.pi)
#  res1 = integralFun1.adaptive(1e-5)
#  print("Integral is: ", res1)
#  print("error:", np.abs(np.pi / 2 - res1))
