"""
Introduction
============

The Munkres module provides an implementation of the Munkres algorithm
(also called the Hungarian algorithm or the Kuhn-Munkres algorithm),
useful for solving the Assignment Problem.

For complete usage documentation, see: https://software.clapper.org/munkres/
"""

__docformat__ = 'markdown'

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__     = ['Munkres', 'make_cost_matrix', 'DISALLOWED']

import numpy as np

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

AnyNum = NewType('AnyNum', Union[int, float])
Matrix = NewType('Matrix', Sequence[Sequence[AnyNum]])

# Info about the module
__version__   = "1.1.4"
__author__    = "Brian Clapper, bmc@clapper.org"
__url__       = "https://software.clapper.org/munkres/"
__copyright__ = "(c) 2008-2020 Brian M. Clapper"
__license__   = "Apache Software License"

# Constants
class DISALLOWED_OBJ(object):
    pass
DISALLOWED = DISALLOWED_OBJ()
DISALLOWED_PRINTVAL = "D"

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UnsolvableMatrix(Exception):
    """
    Exception raised for unsolvable matrices
    """
    pass

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class Munkres:
    """
    Calculate the Munkres solution to the classical assignment problem.
    See the module documentation for usage.
    """

    def __init__(self):
        """Create a new instance"""
        self.C = None
        self.row_covered = []
        self.col_covered = []
        self.n = 0
        self.Z0_r = 0
        self.Z0_c = 0
        self.marked = None
        self.path = None

    def pad_matrix(self, matrix: Matrix, pad_value: int=0) -> Matrix:
        """
        Pad a possibly non-square matrix to make it square.

        **Parameters**

        - `matrix` (list of lists of numbers): matrix to pad
        - `pad_value` (`int`): value to use to pad the matrix

        **Returns**

        a new, possibly padded, matrix
        """
        max_columns = 0
        total_rows = len(matrix)

        for row in matrix:
            max_columns = max(max_columns, len(row))

        total_rows = max(max_columns, total_rows)

        new_matrix = []
        for row in matrix:
            row_len = len(row)
            new_row = row[:]
            if total_rows > row_len:
                # Row too short. Pad it.
                new_row += [pad_value] * (total_rows - row_len)
            new_matrix += [new_row]

        while len(new_matrix) < total_rows:
            new_matrix += [[pad_value] * total_rows]

        return new_matrix


    def calculate_utility(x_i:float, x:np.ndarray, p:np.ndarray, sigma_i: float, epsilon: float) -> float:
        """
        计算效用函数 u_c(x_i) = (x_i / sum(x_n)) * (sum(p_n) - epsilon * sigma_i * x_i)

        参数:
        x_i: float - 当前的 x 值
        x: np.ndarray - 所有 x 值的数组
        : np.ndarray - p_n 值的数组
        sigma_i: float - σi 值
        epsilon: float - ε 值

        返回:
        float: 计算得到的效用值
        """
        # 计算分母部分（所有x的和）
        x_sum = np.sum(x)

        # 计算第一个系数（x_i / sum(x_n)）
        coefficient = x_i / x_sum


        # 计算所有p的和
        p_sum = np.sum(p)

        # 计算第一部分
        part_1 = coefficient * p_sum

        # 计算第二部分
        part_2 = epsilon * sigma_i * x_i

        # 计算效用函数 u_c(x_i)
        utility = part_1 - part_2

        return utility

    def compute(self, cost_matrix: Matrix) -> Sequence[Tuple[int, int]]:
        """计算最优匹配方案

        输入cost_matrix为成本矩阵,返回最优匹配的(行,列)索引对列表
        支持方形和矩形矩阵,不规则矩阵不支持
        非方形矩阵会用0填充成方阵
        """
        # 填充成方阵并初始化
        self.C = self.pad_matrix(cost_matrix)  # 填充矩阵
        self.n = len(self.C)  # 方阵大小
        self.original_length = len(cost_matrix)  # 原矩阵行数
        self.original_width = len(cost_matrix[0])  # 原矩阵列数
        self.row_covered = [False for i in range(self.n)]  # 行覆盖标记
        self.col_covered = [False for i in range(self.n)]  # 列覆盖标记
        self.Z0_r = 0  # 当前零元素行号
        self.Z0_c = 0  # 当前零元素列号
        self.path = self.__make_matrix(self.n * 2, 0)  # 记录交错路径
        self.marked = self.__make_matrix(self.n, 0)  # 标记矩阵(0:无标记 1:星号 2:撇号)

        # 执行匈牙利算法的6个步骤
        done = False
        step = 1
        steps = {1: self.__step1,  # 行归约
                 2: self.__step2,  # 找零元素并标星
                 3: self.__step3,  # 覆盖带星列
                 4: self.__step4,  # 找未覆盖零并标记
                 5: self.__step5,  # 构造交错路径
                 6: self.__step6}  # 调整矩阵

        while not done:
            try:
                func = steps[step]
                step = func()
            except KeyError:
                done = True

        # 提取最优匹配结果
        results = []
        for i in range(self.original_length):
            for j in range(self.original_width):
                if self.marked[i][j] == 1:  # 带星元素即为匹配
                    results += [(i, j)]

        return results

    def __copy_matrix(self, matrix: Matrix) -> Matrix:
        """
        深拷贝矩阵
        深拷贝(deep copy)是指创建一个新的矩阵,不仅复制矩阵本身,还复制其中的所有嵌套数据。
        与之相对的是浅拷贝(shallow copy),只复制最外层结构。
        例如对于矩阵 [[1,2], [3,4]]:
        浅拷贝: 新矩阵引用原始内部列表
        深拷贝: 创建完全独立的新矩阵
        深拷贝确保修改新矩阵不会影响原矩阵。
        """
        return copy.deepcopy(matrix)

    def __make_matrix(self, n: int, val: AnyNum) -> Matrix:
        """创建n*n大小的矩阵,填充指定初值val"""
        matrix = []
        for i in range(n):
            matrix += [[val for j in range(n)]]
        return matrix

    def __step1(self) -> int:
        """
        第一步:行归约 - 从每行中减去该行的最小值,使每行都有零元素

        DISALLOWED表示不可达的权重值，例如：有的工人拒绝前往地点进行服务、某工人不具备完成特定任务的技能

        """
        C = self.C  # 获取成本矩阵的引用
        n = self.n  # 获取矩阵大小

        # 对每一行进行处理
        for i in range(n):
            # 获取该行中所有非禁止值
            vals = [x for x in self.C[i] if x is not DISALLOWED]

            # 如果该行全是禁止值,则无法求解
            if len(vals) == 0:
                raise UnsolvableMatrix("第{0}行全部禁止".format(i))

            # 找出该行最小值
            minval = min(vals)

            # 该行每个非禁止元素都减去最小值
            # 这样确保每行至少有一个0,为后续找最优匹配做准备
            for j in range(n):
                if self.C[i][j] is not DISALLOWED:
                    self.C[i][j] -= minval

        return 2  # 进入第2步

    def __step2(self) -> int:
        """
        第二步:找零元素并标星
        遍历矩阵,对于未被覆盖的0元素,如果其所在行列都没有星号,则标记星号
        """
        n = self.n

        # 遍历矩阵找零元素
        for i in range(n):
            for j in range(n):
                # 找到未覆盖的零元素,且其行列都未标记星号
                if (self.C[i][j] == 0) and \
                        (not self.col_covered[j]) and \
                        (not self.row_covered[i]):
                    self.marked[i][j] = 1  # 标记星号
                    self.col_covered[j] = True  # 覆盖此列
                    self.row_covered[i] = True  # 覆盖此行
                    break  # 找到一个就处理下一行

        self.__clear_covers()  # 清除所有覆盖标记
        return 3  # 进入第3步

    def __step3(self) -> int:
        """
        第三步:覆盖星号列并判断是否完成
        覆盖所有包含星号的列,如果覆盖的列数等于矩阵维度,说明找到完整匹配
        """
        n = self.n
        count = 0  # 记录覆盖的列数

        # 遍历矩阵
        for i in range(n):
            for j in range(n):
                # 找到未覆盖列中的星号元素
                if self.marked[i][j] == 1 and not self.col_covered[j]:
                    self.col_covered[j] = True  # 覆盖该列
                    count += 1  # 覆盖列数+1

        # 判断是否完成匹配
        if count >= n:  # 覆盖列数等于矩阵维度,匹配完成
            step = 7
        else:  # 未完成匹配,继续第4步
            step = 4

        return step

    def __step4(self) -> int:
        """第四步:找未覆盖的零并标记撇号
        找到未覆盖的零,标记撇号
        如果该行没有星号 -> 进入第5步
        否则:覆盖此行,取消覆盖星号所在列,继续找零
        如果没有未覆盖的零 -> 进入第6步"""

        step = 0
        done = False
        row = col = 0
        star_col = -1

        while not done:
            # 找未覆盖的零
            (row, col) = self.__find_a_zero(row, col)

            if row < 0:  # 没找到未覆盖的零
                done = True
                step = 6  # 进入第6步
            else:  # 找到未覆盖的零
                self.marked[row][col] = 2  # 标记撇号
                star_col = self.__find_star_in_row(row)  # 在此行找星号

                if star_col >= 0:  # 找到星号
                    col = star_col
                    self.row_covered[row] = True  # 覆盖此行
                    self.col_covered[col] = False  # 取消覆盖星号列
                else:  # 没找到星号
                    done = True
                    self.Z0_r = row  # 记录当前零元素位置
                    self.Z0_c = col  # 用于第5步
                    step = 5  # 进入第5步

        return step

    def __step5(self) -> int:
        """
        第五步:构造交错路径并调整标记
        从第4步找到的未覆盖撇号零(Z0)开始,构造交错路径:
        Z0(撇号) -> Z1(星号) -> Z2(撇号) -> Z3(星号)...
        直到找到一个其列中没有星号的撇号零为止
        然后将路径上的星号取消,将撇号变为星号
        """

        count = 0  # 路径长度计数
        path = self.path  # 存储交错路径
        # 存入起点Z0(来自第4步找到的未覆盖撇号零)
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c

        done = False
        while not done:
            # 在当前零元素的列中找星号
            row = self.__find_star_in_col(path[count][1])
            if row >= 0:  # 找到星号
                count += 1
                path[count][0] = row  # 记录星号位置
                path[count][1] = path[count - 1][1]  # 保持同列
            else:  # 未找到星号,路径结束
                done = True

            if not done:  # 找到星号,继续找撇号
                col = self.__find_prime_in_row(path[count][0])
                count += 1
                path[count][0] = path[count - 1][0]  # 保持同行
                path[count][1] = col  # 记录撇号位置

        # 根据路径调整标记
        self.__convert_path(path, count)  # 转换路径上的标记
        self.__clear_covers()  # 清除覆盖
        self.__erase_primes()  # 清除所有撇号
        return 3  # 返回第3步

    def __step6(self) -> int:
        """
        第六步:调整矩阵值
        被覆盖行的元素加上最小值
        未覆盖列的元素减去最小值
        保持星号和撇号不变
        """

        # 找出未覆盖元素中的最小值
        minval = self.__find_smallest()
        events = 0  # 记录实际的矩阵变化次数

        for i in range(self.n):
            for j in range(self.n):
                if self.C[i][j] is DISALLOWED:
                    continue  # 跳过禁止项

                # 被覆盖行加最小值
                if self.row_covered[i]:
                    self.C[i][j] += minval
                    events += 1

                # 未覆盖列减最小值
                if not self.col_covered[j]:
                    self.C[i][j] -= minval
                    events += 1

                # 如果一个元素既在覆盖行又在未覆盖列
                # 加减抵消,实际没有变化
                if self.row_covered[i] and not self.col_covered[j]:
                    events -= 2

        # 如果矩阵没有任何实际变化,说明无解
        if (events == 0):
            raise UnsolvableMatrix("Matrix cannot be solved!")

        return 4  # 返回第4步

    def __find_smallest(self) -> AnyNum:
        """Find the smallest uncovered value in the matrix."""
        minval = sys.maxsize
        for i in range(self.n):
            for j in range(self.n):
                if (not self.row_covered[i]) and (not self.col_covered[j]):
                    if self.C[i][j] is not DISALLOWED and minval > self.C[i][j]:
                        minval = self.C[i][j]
        return minval


    def __find_a_zero(self, i0: int = 0, j0: int = 0) -> Tuple[int, int]:
        """Find the first uncovered element with value 0"""
        row = -1
        col = -1
        i = i0
        n = self.n
        done = False

        while not done:
            j = j0
            while True:
                if (self.C[i][j] == 0) and \
                        (not self.row_covered[i]) and \
                        (not self.col_covered[j]):
                    row = i
                    col = j
                    done = True
                j = (j + 1) % n
                if j == j0:
                    break
            i = (i + 1) % n
            if i == i0:
                done = True

        return (row, col)

    def __find_star_in_row(self, row: Sequence[AnyNum]) -> int:
        """
        Find the first starred element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 1:
                col = j
                break

        return col

    def __find_star_in_col(self, col: Sequence[AnyNum]) -> int:
        """
        Find the first starred element in the specified row. Returns
        the row index, or -1 if no starred element was found.
        """
        row = -1
        for i in range(self.n):
            if self.marked[i][col] == 1:
                row = i
                break

        return row

    def __find_prime_in_row(self, row) -> int:
        """
        Find the first prime element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 2:
                col = j
                break

        return col

    def __convert_path(self,
                       path: Sequence[Sequence[int]],
                       count: int) -> None:
        for i in range(count+1):
            if self.marked[path[i][0]][path[i][1]] == 1:
                self.marked[path[i][0]][path[i][1]] = 0
            else:
                self.marked[path[i][0]][path[i][1]] = 1

    def __clear_covers(self) -> None:
        """Clear all covered matrix cells"""
        for i in range(self.n):
            self.row_covered[i] = False
            self.col_covered[i] = False

    def __erase_primes(self) -> None:
        """Erase all prime markings"""
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 2:
                    self.marked[i][j] = 0

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def make_cost_matrix(
        profit_matrix: Matrix,
        inversion_function: Optional[Callable[[AnyNum], AnyNum]] = None
    ) -> Matrix:
    """
    Create a cost matrix from a profit matrix by calling `inversion_function()`
    to invert each value. The inversion function must take one numeric argument
    (of any type) and return another numeric argument which is presumed to be
    the cost inverse of the original profit value. If the inversion function
    is not provided, a given cell's inverted value is calculated as
    `max(matrix) - value`.

    This is a static method. Call it like this:

        from munkres import Munkres
        cost_matrix = Munkres.make_cost_matrix(matrix, inversion_func)

    For example:

        from munkres import Munkres
        cost_matrix = Munkres.make_cost_matrix(matrix, lambda x : sys.maxsize - x)

    **Parameters**

    - `profit_matrix` (list of lists of numbers): The matrix to convert from
       profit to cost values.
    - `inversion_function` (`function`): The function to use to invert each
       entry in the profit matrix.

    **Returns**

    A new matrix representing the inversion of `profix_matrix`.
    """
    if not inversion_function:
      maximum = max(max(row) for row in profit_matrix)
      inversion_function = lambda x: maximum - x

    cost_matrix = []
    for row in profit_matrix:
        cost_matrix.append([inversion_function(value) for value in row])
    return cost_matrix

def print_matrix(matrix: Matrix, msg: Optional[str] = None) -> None:
    """
    Convenience function: Displays the contents of a matrix.

    **Parameters**

    - `matrix` (list of lists of numbers): The matrix to print
    - `msg` (`str`): Optional message to print before displaying the matrix
    """
    import math

    if msg is not None:
        print(msg)

    # Calculate the appropriate format width.
    width = 0
    for row in matrix:
        for val in row:
            if val is DISALLOWED:
                val = DISALLOWED_PRINTVAL
            width = max(width, len(str(val)))

    # Make the format string
    format = ('%%%d' % width)

    # Print the matrix
    for row in matrix:
        sep = '['
        for val in row:
            if val is DISALLOWED:
                val = DISALLOWED_PRINTVAL
            formatted = ((format + 's') % val)
            sys.stdout.write(sep + formatted)
            sep = ', '
        sys.stdout.write(']\n')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    matrices = [
        # Square
        ([[400, 150, 400],
          [400, 450, 600],
          [300, 225, 300]],
         850),  # expected cost

        # Rectangular variant
        ([[400, 150, 400, 1],
          [400, 450, 600, 2],
          [300, 225, 300, 3]],
         452),  # expected cost


        # Square
        ([[10, 10,  8],
          [9,  8,  1],
          [9,  7,  4]],
         18),

        # Square variant with floating point value
        ([[10.1, 10.2,  8.3],
          [9.4,  8.5,  1.6],
          [9.7,  7.8,  4.9]],
         19.5),

        # Rectangular variant
        ([[10, 10,  8, 11],
          [9,  8,  1, 1],
          [9,  7,  4, 10]],
         15),

        # Rectangular variant with floating point value
        ([[10.01, 10.02,  8.03, 11.04],
          [9.05,  8.06,  1.07, 1.08],
          [9.09,  7.1,  4.11, 10.12]],
         15.2),

        # Rectangular with DISALLOWED
        ([[4, 5, 6, DISALLOWED],
          [1, 9, 12, 11],
          [DISALLOWED, 5, 4, DISALLOWED],
          [12, 12, 12, 10]],
         20),

        # Rectangular variant with DISALLOWED and floating point value
        ([[4.001, 5.002, 6.003, DISALLOWED],
          [1.004, 9.005, 12.006, 11.007],
          [DISALLOWED, 5.008, 4.009, DISALLOWED],
          [12.01, 12.011, 12.012, 10.013]],
         20.028),

        # DISALLOWED to force pairings
        ([[1, DISALLOWED, DISALLOWED, DISALLOWED],
          [DISALLOWED, 2, DISALLOWED, DISALLOWED],
          [DISALLOWED, DISALLOWED, 3, DISALLOWED],
          [DISALLOWED, DISALLOWED, DISALLOWED, 4]],
         10),

        # DISALLOWED to force pairings with floating point value
        ([[1.1, DISALLOWED, DISALLOWED, DISALLOWED],
          [DISALLOWED, 2.2, DISALLOWED, DISALLOWED],
          [DISALLOWED, DISALLOWED, 3.3, DISALLOWED],
          [DISALLOWED, DISALLOWED, DISALLOWED, 4.4]],
         11.0)]

    m = Munkres()
    for cost_matrix, expected_total in matrices:
        print_matrix(cost_matrix, msg='cost matrix')
        indexes = m.compute(cost_matrix)
        total_cost = 0
        for r, c in indexes:
            x = cost_matrix[r][c]
            total_cost += x
            print(('(%d, %d) -> %s' % (r, c, x)))
        print(('lowest cost=%s' % total_cost))
        assert expected_total == total_cost
