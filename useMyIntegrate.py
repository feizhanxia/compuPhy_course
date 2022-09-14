import numpy as np
from myIntegrate import simpson


def fun(x):
    return (np.sin(x)**2)


integralFun = simpson(fun, 0, np.pi)
res = integralFun.adaptive(1e-5)
print("Integral is: ", res)
print("error:", np.abs(np.pi / 2 - res))
