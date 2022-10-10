import numpy as np
import time
from numba import jit  # numba即时编译加速与numpy有关的函数，暂时对类的支持不好，不支持子类，在这里只采取函数式编程
import sys
import matplotlib.pyplot as plt
from matplotlib import ticker
from docx import Document

#  rng = np.random.default_rng()
# prepare for visualization
integrateValue = []
integrateError = []


@jit(nopython=True, parallel=True)
def mcIntegrate(func, a, b, n):  # 区间a到b上n个点的蒙卡积分
    between = b - a
    x1 = np.random.random((n))
    ans = func(x1 * between + a).sum() * (b - a) / n
    return ans


ufuncMcIntegrate = np.frompyfunc(
    mcIntegrate, 4, 1)  # 把区间a到b上n个点的蒙卡积分转化为numpy的ufunc即对于np.array广播

pointNum = 100000  # 手动设置每个区间的点数


# 参考Mathematica中NIntegrate函数的AdaptiveMonteCarlo方法，采用分区间蒙卡提高精度
def adaptive(func, a, b, eps):
    it = 0
    maxIt = 10000000
    error = sys.float_info.max
    nodeCount = 2
    value1 = 0
    value2 = 0
    while error > eps and maxIt > it:
        nodeCount = nodeCount * 2
        nodes1 = np.linspace(a, b, nodeCount)  # 均匀划分区间以在每个区间进行蒙卡
        nodes2 = np.linspace(a, b, nodeCount * 2)
        # 广播到区间进行蒙卡，并求和。单区间蒙卡点数手动设置为pointNum，需要手动设置是本算法的缺陷。
        value1 = ufuncMcIntegrate(funcToIntegrate, nodes1[:-1], nodes1[1:],
                                  pointNum).sum()
        value2 = ufuncMcIntegrate(funcToIntegrate, nodes2[:-1], nodes2[1:],
                                  pointNum).sum()
        error = abs(value1 - value2)
        print(error)
        integrateValue.append(value2)
        integrateError.append(error)
        it += 1
    return value2


@jit(nopython=True, parallel=True)
def funcToIntegrate(x):  # 定义可以广播到array的待积分函数
    return x**2 + np.cos(x + 2)


#  def I0(x): # 正比例函数用来验证简单情况
#  return x

if __name__ == '__main__':
    t0 = time.time()
    answer = adaptive(funcToIntegrate, -2, 3, 1e-6)
    t1 = time.time()
    print("ans=", answer, "time=", t1 - t0)
    # 下面输出word格式的收敛表格
    document = Document()
    table = document.add_table(1, 4)
    heading_cells = table.rows[0].cells
    heading_cells[0].text = 'Iteration'
    heading_cells[1].text = 'Total number of points'
    heading_cells[2].text = 'Result'
    heading_cells[3].text = 'Error'
    for i, val in enumerate(np.array(integrateValue)):
        cells = table.add_row().cells
        cells[0].text = str(i)
        cells[1].text = str(2**(i + 1) * pointNum)
        cells[2].text = '%.9g' % val
        cells[3].text = '%.4e' % integrateError[i]
    table.style = 'Light Shading Accent 1'
    document.save('1-1.docx')
    # 下面开始画图
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    X = range(len(integrateError))
    Y1 = np.log10(np.array(integrateError))
    Y2 = np.array(integrateValue)
    axs[0].plot(X, Y1, color="C4", linewidth=1, marker=".")
    axs[0].xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0))
    axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('log10(error)')
    axs[0].set_title("Error-Iteration Plot")
    axs[1].plot(X, Y2, color="C0", linewidth=1, marker=".")
    axs[1].xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0))
    axs[1].yaxis.set_major_locator(ticker.AutoLocator())
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Integration Value')
    axs[1].set_title("Value-Iteration Plot")
    axs[1].plot(X,
                np.full_like(Y2, integrateValue[-1]),
                color="C1",
                linestyle="--")
    fig.savefig("1-1.png", dpi=300)
    plt.show()

#  print(ufuncMcIntegrate(funcToIntegrate, np.array([1, 2, 3]),np.array([6, 5, 4]), 1000))
#  print(funcToIntegrate(3))
# 验证numpy矩阵运算自动cpu并行化
#  n = 20000
#  A = np.random.randn(n, n).astype('float64')
#  B = np.random.randn(n, n).astype('float64')
#  start_time = time.time()
#  nrm = np.linalg.norm(A @ B)
#  print(" took {} seconds ".format(time.time() - start_time))
#  print(" norm = ", nrm)
#  print(np.__config__.show())
