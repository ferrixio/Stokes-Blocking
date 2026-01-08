import numpy as np
from numpy import save, load, loadtxt
from math import exp, ceil, sqrt
from random import shuffle
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.fft import fft, ifft
from scipy.sparse.linalg import gmres
from scipy.linalg import hankel, lu
from numpy.linalg import pinv, inv, eigh, eig
from time import time

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


class krylov_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))
            if rk < 1e-5:
                quit()


class StiffElement:
    '''Object that stores informations to build the stiffness matrix

    ATTRIBUTES:
    - i [int]                    = row index in the big stiffness matrix
    - j [int]                    = column index in the big stiffness matrix
    - mu [float]                 = float to assign at the position (i,j)
    - stiffIndex [list[int,int]] = relative position during the construction of the elements
    '''
    def __init__(self, i:int, j:int, item:float, stiffIndex:list=[]) -> None:
        self.i = i
        self.j = j
        self.mu = item
        self.stiffIndex = stiffIndex

    def __str__(self) -> str:
        '''Magic method to print the element in a pretty way using print()'''
        return f"({self.i},{self.j}) -> entry: {self.mu} -> stiffIndex {self.stiffIndex}"


# Funzioni generatrici della matrice A_xx
def ZeroColumn(n:int, mu:list[float]) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the global stiffness matrix
    at the end - Zero column
    """
    elem_even, elem_odd = [], []

    # even part
    for k1 in range(n):
        i = 2*k1
        elem_even.extend([
            StiffElement(i, i, 8/3 * (mu[4*k1]+mu[1+4*k1])),
            StiffElement(i, i+2*n, -2/3 * (mu[4*k1]+mu[1+4*k1]))
        ])
        if 0!=k1:
            elem_even.append(StiffElement(i, i+2*n-1, -4/3*mu[1+4*k1]))

    # odd part
    for k1 in range(n):
        i = 1+2*k1
        elem_odd.extend([
            StiffElement(i, i, 8/3 * (mu[4*k1]+mu[2+4*k1])),
            StiffElement(i, i+2*n-1, -2/3 * (mu[4*k1]+mu[2+4*k1]))
        ])
        if (n-1)!=k1:
            elem_odd.append(StiffElement(i, i+2*n, -4/3*mu[2+4*k1]))

    return elem_even, elem_odd

def SecondColumn(n:int, mu:list[float]) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the global stiffness matrix
    at the end - Second + (n-1) columns"""
    even_rows, odd_rows = [], []

    # even part
    for k1 in range(n-1):
        for k2 in range(n):
            i = 4*n-1 + (8*n-2)*k1 + 2*k2
            even_rows.extend([
                StiffElement(i, i-2*n+1, -2/3 * (mu[1+4*n*k1+4*k2]+mu[3+4*n*k1+4*k2])),
                StiffElement(i, i, 8/3 * (mu[1+4*n*k1+4*k2]+mu[3+4*n*k1+4*k2])),
                StiffElement(i, i+2*n, -4/3*mu[3+4*n*k1+4*k2])]
            )

            if 0!=k2:
                even_rows.extend([
                    StiffElement(i,i-2*n, -4/3*mu[1+4*n*k1+4*k2]),
                    StiffElement(i,i+2*n-1, -2/3 * (mu[1+4*n*k1+4*k2]+mu[3+4*n*k1+4*k2]))]
                    )

    # odd part
    for k1 in range(n-1):
        for k2 in range(n):
            i = 4*n + (8*n-2)*k1 + 2*k2
            odd_rows.extend([
                StiffElement(i, i-2*n, -2/3 * (mu[2+4*n*k1+4*k2]+mu[3+4*n*k1+4*k2])),
                StiffElement(i, i, 8/3 * (mu[2+4*n*k1+4*k2]+mu[3+4*n*k1+4*k2])),
                StiffElement(i, i+2*n-1, -4/3*mu[3+4*n*k1+4*k2])]
            )

            if (n-1)!=k2:
                odd_rows.extend([
                    StiffElement(i,i-2*n+1, -4/3*mu[2+4*n*k1+4*k2]),
                    StiffElement(i,i+2*n, -2/3 * (mu[2+4*n*k1+4*k2]+mu[3+4*n*k1+4*k2]))]
                )

    return even_rows, odd_rows

def Centers(n:int, mu:list[float]) -> tuple[list[StiffElement], list[StiffElement], list[StiffElement], list[StiffElement]]:
    """Returns the evaluations in the centers of cells"""
    bottom_row, middle_bottom, middle_top, top_row = [], [], [], []
    small = 8*n-2

    #bottom part
    for k1 in range(n):
        i = 2*n + small*k1
        base = 4*n*k1
        bottom_row.extend([
            StiffElement(i, i-2*n, -2/3 * (mu[base]+mu[1+base])),
            StiffElement(i, i-2*n+1, -2/3 * (mu[base]+mu[2+base])),
            StiffElement(i, i, mu[base]+mu[1+base]+mu[2+base]+mu[3+base]),
            StiffElement(i, i+2*n-1, -2/3 * (mu[1+base]+mu[3+base])),
            StiffElement(i, i+2*n, -2/3 * (mu[2+base]+mu[3+base]))]
        )

        if k1 != 0:
            bottom_row.append(
                StiffElement(i, i-4*n+2, 1/6 * (mu[base]+mu[2+base]))
            )

        if k1 != (n-1):
            bottom_row.append(
                StiffElement(i, i+4*n, 1/6 * (mu[2+base]+mu[3+base]))
            )

    # middle bottom part
    for k1 in range(n):
        for k2 in range(n-1):
            i = 2*n+1 + small*k1 + 2*k2
            base = 4*n*k1+4*k2
            middle_bottom.extend([
                StiffElement(i, i-2*n, -4/3*mu[2+base]),
                StiffElement(i, i-2*n+1, -4/3*mu[5+base]),
                StiffElement(i, i, 8/3 * (mu[2+base]+mu[5+base])),
                StiffElement(i, i+2*n-1, -4/3*mu[2+base]),
                StiffElement(i, i+2*n, -4/3*mu[5+base])]
            )

    # middle top part
    for k1 in range(n):
        for k2 in range(1,n-1):
            i = 2*n + small*k1 + 2*k2
            base = 4*n*k1+4*k2
            middle_top.extend([
                StiffElement(i, i-2*n, -2/3 * (mu[base]+mu[1+base])),
                StiffElement(i, i-2*n+1, -2/3 * (mu[base]+mu[2+base])),
                StiffElement(i, i, (mu[base]+mu[1+base]+mu[2+base]+mu[3+base])),
                StiffElement(i, i+2*n-1, -2/3 * (mu[1+base]+mu[3+base])),
                StiffElement(i, i+2*n, -2/3 * (mu[2+base]+mu[3+base]))]
            )

            if k1 != 0:
                middle_top.extend([
                    StiffElement(i, i-4*n, 1/6 * (mu[base]+mu[1+base])),
                    StiffElement(i, i-4*n+2, 1/6 * (mu[base]+mu[base+2]))]
                )

            if k1 != (n-1):
                middle_top.extend([
                    StiffElement(i, i+4*n-2, 1/6 * (mu[1+base]+mu[3+base])),
                    StiffElement(i, i+4*n, 1/6 * (mu[2+base]+mu[3+base]))]
                )

    # top row part
    for k1 in range(n):
        i = 4*n-2 + small*k1
        base = 4*n*k1 + 4*(n-1)
        top_row.extend([
            StiffElement(i, i-2*n, -2/3 * (mu[base]+mu[1+base])),
            StiffElement(i, i-2*n+1, -2/3 * (mu[base]+mu[2+base])),
            StiffElement(i, i,(mu[base]+mu[1+base]+mu[2+base]+mu[3+base])),
            StiffElement(i, i+2*n-1, -2/3 * (mu[1+base]+mu[3+base])),
            StiffElement(i, i+2*n, -2/3 * (mu[2+base]+mu[3+base]))]
        )

        if k1 != 0:
            top_row.append(
                StiffElement(i, i-4*n, 1/6 * (mu[base]+mu[1+base]))
            )

        if k1 != (n-1):
            top_row.append(
                StiffElement(i, i+4*n-2, 1/6 * (mu[1+base]+mu[3+base]))
            )

    return bottom_row, middle_bottom, middle_top, top_row

def ThirdColumn(n:int, mu:list[float]) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the global stiffness matrix
    at the end - Third + (n-1) columns"""
    even_rows, odd_rows = [], []
    small = 8*n-2

    # even rows
    for k1 in range(n-1):
        for k2 in range(n):
            i = 6*n-1 + small*k1 + 2*k2
            base = 4*n*k1+4*k2

            even_rows.extend([
                StiffElement(i, i-2*n, -4/3*mu[3+base]),
                StiffElement(i, i-2*n+1, -4/3*mu[3+base]),
                StiffElement(i, i, 8/3*(mu[3+base]+mu[4*n+base])),
                StiffElement(i, i+2*n-1, -4/3*mu[4*n+base]),
                StiffElement(i, i+2*n, -4/3*mu[4*n+base])]
            )

    # odd rows
    for k1 in range(n-1):
        for k2 in range(n-1):
            i = 6*n + small*k1 + 2*k2
            base = 4*n*k1+4*k2
           
            odd_rows.extend([
                StiffElement(i, i-4*n, 1/6 * (mu[2+base]+mu[3+base])),
                StiffElement(i, i-4*n+2, 1/6 * (mu[5+base]+mu[7+base])),
                StiffElement(i, i-2*n, -2/3 * (mu[2+base]+mu[3+base])),
                StiffElement(i, i-2*n+1, -2/3 * (mu[5+base]+mu[7+base])),
                StiffElement(i, i, 1/2 * (mu[2+base]+mu[3+base]+mu[5+base]+mu[7+base]+mu[4*n+base]+mu[4*n+2+base]+mu[4*n+4+base]+mu[4*n+5+base])),
                StiffElement(i, i+2*n-1, -2/3 * (mu[4*n+base]+mu[4*n+2+base])),
                StiffElement(i, i+2*n, -2/3 * (mu[4*n+4+base]+mu[4*n+5+base])),
                StiffElement(i, i+4*n-2, 1/6 * (mu[4*n+base]+mu[4*n+2+base])),
                StiffElement(i, i+4*n, 1/6 * (mu[4*n+4+base]+mu[4*n+5+base]))]
            )

    return even_rows, odd_rows

def FourthColumn(n:int, mu:list[float]) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the global stiffness matrix
    at the end - Fourth + (n-1) columns"""
    even_rows, odd_rows = [], []
    small = 8*n-2

    # even part
    for k1 in range(1, n):
        for k2 in range(n):
            i = small*k1 + 2*k2
            base = 4*n*k1+4*k2

            even_rows.extend([
                StiffElement(i, i-2*n+1, -4/3*mu[base], stiffIndex=(k1,k2,1)),
                StiffElement(i, i, 8/3*(mu[base]+mu[1+base]), stiffIndex=(k1,k2,2)),
                StiffElement(i, i+2*n, -2/3*(mu[base]+mu[1+base]), stiffIndex=(k1,k2,3))]
            )

            if k2 != 0:
                even_rows.extend([
                    StiffElement(i, i-2*n, -2/3 * (mu[base]+mu[1+base]), stiffIndex=(k1,k2,'d1')),
                    StiffElement(i, i+2*n-1, -4/3 * (mu[1+base]), stiffIndex=(k1,k2,'d2'))]
                )

    # odd part
    for k1 in range(1, n):
        for k2 in range(n):
            i = 1 + small*k1 + 2*k2
            base = 4*n*k1+4*k2

            odd_rows.extend([
                StiffElement(i, i-2*n, -4/3*mu[base]),
                StiffElement(i, i, 8/3 * (mu[base]+mu[2+base])),
                StiffElement(i, i+2*n-1, -2/3 * (mu[base]+mu[2+base]))]
            )

            if k2 != (n-1):
                odd_rows.extend([
                    StiffElement(i, i-2*n+1, -2/3 * (mu[base]+mu[2+base])),
                    StiffElement(i, i+2*n, -4/3 * mu[2+base])]
                )

    return even_rows, odd_rows

def LastColumn(n:int, mu:list[float]) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the global stiffness matrix
    at the end - last column"""
    even_rows, odd_rows = [], []
    small = 8*(n**2)-6*n+1

    # even part
    for k1 in range(n):
        i = small + 2*k1
        base = 4*n*(n-1)+4*k1

        even_rows.extend([
            StiffElement(i, i-2*n+1, -2/3 * (mu[1+base]+mu[3+base])),
            StiffElement(i, i, 8/3 * (mu[1+base]+mu[3+base]))]
        )

        if k1 != 0:
            even_rows.append(StiffElement(i, i-2*n, -4/3 * (mu[1+base])))

    # odd part
    for k1 in range(n):
        i = small +2*k1+1
        base = 4*n*(n-1)+4*k1

        odd_rows.extend([
            StiffElement(i, i-2*n, -2/3 * (mu[2+base]+mu[3+base])),
            StiffElement(i, i, 8/3 * (mu[2+base]+mu[3+base]))]
        )

        if k1 != (n-1):
            odd_rows.append(StiffElement(i, i-2*n+1, -4/3*mu[2+base]))

    return even_rows, odd_rows


def AssembleStiffness(n:int, *arg) -> NDArray:
    """Function that build A_xx of the stiffness matrix from a list of Stiffelements"""
    Nv = 8*(n**2)-4*n+1     # size of the matrix
    A_xx = np.zeros((Nv, Nv))

    res = []
    for item in arg:
        for stiff in item:
            # if the position is empty, fill it
            if not A_xx[stiff.i, stiff.j]:
                A_xx[stiff.i, stiff.j] = stiff.mu
                continue

            # else there is a residual and append to an external list
            res.append([stiff.__str__(), f"item {arg.index(item)}, stiff index {item.index(stiff)}"])

    return A_xx, res


# Funzioni generatrici della matrice B_x
def ZeroColumnBx(n:int) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the Bx matrix - Zero column"""
    even_rows, odd_rows = [], []

    # even part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1 + 2*k2
            # print(f"zero even cols k1={k1} k2={k2}\ti={i}")
            even_rows.extend([
                StiffElement(i, (2*n+1)*k1+k2, -1/6),
                StiffElement(i, (2*n+1)*k1+k2+1, -1/12),
                StiffElement(i, (2*n+1)*k1+1+n+k2, 1/6),
                StiffElement(i, (2*n+1)*k1+1+2*n+k2, 1/12)]
            )

    # odd part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1+1+2*k2
            # print(f"zero odd cols k1={k1} k2={k2}\ti={i}")
            odd_rows.extend([
                StiffElement(i, (2*n+1)*k1+k2, -1/12),
                StiffElement(i, (2*n+1)*k1+1+k2, -1/6),
                StiffElement(i, (2*n+1)*k1+1+n+k2, 1/6),
                StiffElement(i, (2*n+1)*k1+2+2*n+k2, 1/12)]
            )

    return even_rows, odd_rows

def FirstColumnBx(n:int) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the Bx matrix - First + (n-1) columns"""
    odd_rows = []

    # odd part
    for k1 in range(n):
        for k2 in range(n-1):
            i = 2*n+1 + (8*n-2)*k1 + 2*k2
            # print(f"First odd cols k1={k1} k2={k2}\ti={i}")
            odd_rows.extend([
                StiffElement(i, (2*n+1)*k1+1+k2, -1/6),
                StiffElement(i, (2*n+1)*k1+(2*n+2)+k2, 1/6),
            ])

    return odd_rows

def SecondColumnBx(n:int) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the Bx matrix - Second + (n-1) columns"""
    even_rows, odd_rows = [], []

    # even part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1 +(4*n-1)+ 2*k2
            # print(f"Second even cols k1={k1} k2={k2}\ti={i}")
            even_rows.extend([
                StiffElement(i, (2*n+1)*k1+k2, -1/12),
                StiffElement(i, (2*n+1)*k1+k2+n+1, -1/6),
                StiffElement(i, (2*n+1)*k1+(2*n+1)+k2, 1/6),
                StiffElement(i, (2*n+1)*k1+(2*n+2)+k2, 1/12),
            ])

    # odd part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1 + 4*n + 2*k2
            # print(f"Second odd cols k1={k1} k2={k2}\ti={i}")
            odd_rows.extend([
                StiffElement(i, (2*n+1)*k1+1+k2, -1/12),
                StiffElement(i, (2*n+1)*k1+n+1+k2, -1/6),
                StiffElement(i, (2*n+1)*k1+(2*n+1)+k2, 1/12),
                StiffElement(i, (2*n+1)*k1+(2*n+2)+k2, 1/6),
            ])

    return even_rows, odd_rows

def FirstColumnn1Bx(n:int) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the Bx matrix - First + (n-1) columns"""
    even_rows = []

    # even rows
    for k1 in range(n-1):
        for k2 in range(n):
            i = (8*n-2)*k1 + 6*n-1 + 2*k2
            # print(f"FirstColumnn1 even cols k1={k1} k2={k2}\t8/3 * ({3}+{4*n})\ti={i}")
            even_rows.extend([
                StiffElement(i, (2*n+1)*k1+n+1+k2, -1/6),
                StiffElement(i, (2*n+1)*k1+(3*n+2)+k2, 1/6),
            ])

    return even_rows

# Funzioni generatrici della matrice B_y
def ZeroColumnBy(n:int) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the By matrix - Zero column"""
    even_rows, odd_rows = [], []

    # even part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1 + 2*k2
            # print(f"zero even cols k1={k1} k2={k2}\ti={i}")
            even_rows.extend([
                StiffElement(i, (2*n+1)*k1+k2, -1/6),
                StiffElement(i, (2*n+1)*k1+k2+1, 1/12),
                StiffElement(i, (2*n+1)*k1+1+n+k2, 1/6),
                StiffElement(i, (2*n+1)*k1+1+2*n+k2, -1/12)]
            )

    # odd part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1 + 1 + 2*k2
            # print(f"zero odd cols k1={k1} k2={k2}\ti={i}")
            odd_rows.extend([
                StiffElement(i, (2*n+1)*k1+k2, -1/12),
                StiffElement(i, (2*n+1)*k1+k2+1, 1/6),
                StiffElement(i, (2*n+1)*k1+1+n+k2, -1/6),
                StiffElement(i, (2*n+1)*k1+2+2*n+k2, 1/12)]
            )

    return even_rows, odd_rows

def FirstColumnBy(n:int) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the By matrix - First + (n-1) columns"""
    odd_rows = []

    # odd part
    for k1 in range(n):
        for k2 in range(n-1):
            i = 2*n+1 + (8*n-2)*k1 + 2*k2
            # print(f"First odd cols k1={k1} k2={k2}\ti={i}")
            odd_rows.extend([
                StiffElement(i, (2*n+1)*k1+n+1+k2, -1/6),
                StiffElement(i, (2*n+1)*k1+n+2+k2, 1/6),
            ])

    # NOTE: the even part is zero

    return odd_rows

def SecondColumnBy(n:int) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the By matrix - Second + (n-1) columns"""
    even_rows, odd_rows = [], []

    # even part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1 +(4*n-1)+ 2*k2
            # print(f"Second even cols k1={k1} k2={k2}\ti={i}")
            even_rows.extend([
                StiffElement(i, (2*n+1)*k1+k2, -1/12),
                StiffElement(i, (2*n+1)*k1+k2+n+1, 1/6),
                StiffElement(i, (2*n+1)*k1+(2*n+1)+k2, -1/6),
                StiffElement(i, (2*n+1)*k1+(2*n+2)+k2, 1/12),
            ])

    # odd part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1 + 4*n + 2*k2
            # print(f"Second odd cols k1={k1} k2={k2}\ti={i}")
            odd_rows.extend([
                StiffElement(i, (2*n+1)*k1+1+k2, 1/12),
                StiffElement(i, (2*n+1)*k1+n+1+k2, -1/6),
                StiffElement(i, (2*n+1)*k1+(2*n+1)+k2, -1/12),
                StiffElement(i, (2*n+1)*k1+(2*n+2)+k2, 1/6),
            ])

    return even_rows, odd_rows

def FirstColumnn1By(n:int) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the By matrix - First + (n-1) columns"""
    even_rows = []

    # even rows
    for k1 in range(n-1):
        for k2 in range(n):
            i = (8*n-2)*k1 + 6*n-1 + 2*k2
            # print(f"FirstColumnn1 even cols k1={k1} k2={k2}\t8/3 * ({3}+{4*n})\ti={i}")
            even_rows.extend([
                StiffElement(i, (2*n+1)*k1+(2*n+1)+k2, -1/6),
                StiffElement(i, (2*n+1)*k1+(2*n+2)+k2, 1/6),
            ])

    # NOTE: the odd part is zero

    return even_rows


def AssembleB_block(n:int, *arg):
    """Function that build the B_x matrix"""
    Nv = 8*(n**2)-4*n+1     # size of the matrix

    # Determine the maximum j index from all Bx_elements to set Nq appropriately
    max_j = max(stiff.j for item in arg for stiff in item)
    Nq = max_j + 1
    Bx = np.zeros((Nv, Nq));

    res = []
    for item in arg:
        for stiff in item:
            # print(stiff)
            # if the position is empty, fill it
            if not Bx[stiff.i, stiff.j] :
                Bx[stiff.i, stiff.j] = stiff.mu
                continue

            # else there is a residual and append to an external list
            res.append([str(stiff), f"item {arg.index(item)}, stiff index {item.index(stiff)}"])

    return Bx, res


# Generatori delle cose
def CreateMu(n:int, case:int, gamma:int) -> list[float]:
    '''Creates the array of mu coefficients along the grid'''
    ans = []
    grid = np.linspace(0,1,n)

    for i in grid:              # righe
        for j in grid:          # colonne
            for _ in range(4):  # triangoli interni
                match case:
                    case 1:                     # constant
                        ans.append(1)

                    case 2:                     # smooth
                        ans.append(i*j + exp(i+j))

                    case 3:                     # step function
                        ans.append(gamma*(i<=.5 and j<=.5) + (1+i+j)*(i>.5 or j>.5))

                    case _:
                        raise ValueError("Invalid case value")

    return ans

def CreateUniformSampling(mat_shape_row:int, n_orig:int) -> list[float]:
    '''Creates the rhs with an uniform sampling.
    It cuts out the first elements to fit the matrix size'''
    n_ceil = ceil(sqrt(mat_shape_row))
    grid = np.linspace(-np.pi, np.pi, n_ceil)
    ans = [i*j + exp(i+j) for i in grid for j in grid]

    return pow(1/n_orig, 2)*np.array(ans[len(ans) - mat_shape_row:], dtype=float)

def CreateRandomSampling(mat_shape_row:int, n_orig:int) -> list[float]:
    '''Creates the rhs with a random sampling'''
    n_ceil = ceil(sqrt(mat_shape_row))
    grid = np.linspace(-np.pi, np.pi, n_ceil)
    ans = [i*j + exp(i+j) for i in grid for j in grid]
    shuffle(ans)

    return pow(1/n_orig, 2)*np.array(ans[len(ans) - mat_shape_row:], dtype=float)


def Create_Axx_Precond(A_xx:NDArray, n:int) -> NDArray:
    '''Builds the preconditioner for the matrix Axx'''

    Nv = 8*(n**2)-4*n+1         # size of the matrix

    # Estrazione dei blocchi
    D_list, B_list, C_list = [], [], []
    n_diag = (2*n, 2*n)
    n_D_mini = (2*n-2, 2*n-2)
    n_B_mini = (2*n, 2*n-2)
    n_C_odd = (2*n-2, 2*n)
    n_C_even = (2*n, 2*n-2)

    # blocchi diagonali
    lineCounter = 0
    block_index = 1
    while lineCounter < Nv:
        # blocco diagonale piccolo
        if block_index % 4 == 0:
            b = A_xx[lineCounter:lineCounter+n_D_mini[0], lineCounter:lineCounter+n_D_mini[1]]
            lineCounter += n_D_mini[0]

        else:
            b = A_xx[lineCounter:lineCounter+n_diag[0], lineCounter:lineCounter+n_diag[1]]
            lineCounter += n_diag[0]

        D_list.append(b)        
        block_index += 1


    # blocchi B_i
    lineCounter_x = 2*n
    lineCounter_y = 0
    block_index = 1
    while lineCounter_x < Nv:
        # blocco diagonale piccolo
        if block_index % 4 == 0:
            b = A_xx[lineCounter_x:lineCounter_x+n_B_mini[0], lineCounter_y:lineCounter_y+n_B_mini[1]]
            lineCounter_x += n_B_mini[0]
            lineCounter_y += n_B_mini[1]

        # blocco a sinistra del blocco diagonale piccolo
        elif block_index % 4 == 3:
            b = A_xx[lineCounter_x:lineCounter_x+n_B_mini[1], lineCounter_y:lineCounter_y+n_B_mini[0]]
            lineCounter_x += n_B_mini[1]
            lineCounter_y += n_B_mini[0]

        else:
            b = A_xx[lineCounter_x:lineCounter_x+n_diag[0], lineCounter_y:lineCounter_y+n_diag[1]]
            lineCounter_x += n_diag[1]
            lineCounter_y += n_diag[0]

        B_list.append(b)        
        block_index += 1


    # blocchi C_i
    lineCounter_x = 4*n
    lineCounter_y = 0
    block_index = 1
    c_index = 1
    while lineCounter_x < Nv:
        if block_index % 2 != 0:
            b = A_xx[lineCounter_x:lineCounter_x+n_diag[0], lineCounter_y:lineCounter_y+n_diag[1]]
            lineCounter_x += n_diag[0]
            lineCounter_y += n_diag[1]

        elif c_index % 2 == 0:
            b = A_xx[lineCounter_x:lineCounter_x+n_C_even[0], lineCounter_y:lineCounter_y+n_C_even[1]]
            lineCounter_x += n_C_even[1]
            lineCounter_y += n_C_even[0]
            c_index += 1

        elif c_index % 2 == 1:
            b = A_xx[lineCounter_x:lineCounter_x+n_C_odd[0], lineCounter_y:lineCounter_y+n_C_odd[1]]
            lineCounter_x += n_C_odd[1]
            lineCounter_y += n_C_odd[0]
            c_index += 1

        C_list.append(b)        
        block_index += 1


    # calcolo dei precondizionatori delle sequenze B e C
    prec_B = precond_tau_rect(B_list, n_diag, n)
    prec_C = precond_tau_rect(C_list, n_diag, n)

    # assemblaggio del precondizionatore
    ansD = D_list[0]
    for i in range(1,len(D_list)):
        ansD = np.block([
            [ansD, np.zeros((ansD.shape[0], D_list[i].shape[1]))],
            [np.zeros((D_list[i].shape[0], ansD.shape[1])), D_list[i]]
        ])
    
    ansB = np.block([[np.zeros((n_diag))],[prec_B[0]]])
    for i in range(1, len(prec_B)):
        ansB = np.block([
            [ansB, np.zeros((ansB.shape[0], prec_B[i].shape[1]))],
            [np.zeros((prec_B[i].shape[0], ansB.shape[1])), prec_B[i]]
        ])
    ansB = np.hstack((ansB, np.zeros((ansB.shape[0], D_list[-1].shape[1]))))

    ansC = np.block([[np.zeros((n_diag))],[np.zeros(prec_B[0].shape)],[prec_C[0]]])
    for i in range(1, len(prec_C)):
        ansC = np.block([
            [ansC, np.zeros((ansC.shape[0], prec_C[i].shape[1]))],
            [np.zeros((prec_C[i].shape[0], ansC.shape[1])), prec_C[i]]
        ])
    ansC = np.hstack((ansC, np.zeros((ansC.shape[0], prec_B[-1].shape[1])), np.zeros((ansC.shape[0], D_list[-1].shape[1]))))

    # assembla e ritorna A con già Axx e Ayy
    return ansD + ansB + ansC + ansB.T + ansC.T

def Create_B_Precond(Bx:NDArray, By:NDArray, n:int) -> NDArray:
    '''Builds the unilevel preconditioner for B'''
    # precondizionatore di B_x
    # estrazione dei blocchi di Bx
    L_list = [
        Bx[:2*n, :n+1],
        Bx[2*n:4*n-1, :n+1],
        Bx[4*n-1:6*n-1, :n+1],
        Bx[4*n-1:6*n-1, n+1:2*n+1],
        Bx[6*n-1:8*n-2, n+1:2*n+1]
    ]

    # calcolo dei precondizionatori di Bx
    pL134 = precond_circ_inplace([L_list[0], L_list[2], L_list[3]], (2*n, n), n)
    pL2J = precond_circ_inplace([L_list[1], L_list[4]], (2*n-2, n-1), n-1)

    LL1 = np.block([
        [pL134[0], -pL134[2]],
        [pL2J[0], np.zeros((2*n-1, n))],
        [pL134[1], pL134[2]]
    ])

    LL2 = np.block([
        [-pL134[1], np.zeros((2*n, n))],
        [-pL2J[0], np.zeros((2*n-1, n))],
        [-pL134[0], np.zeros((2*n, n))]
    ])

    LL3 = np.hstack((np.zeros((2*n-1, n+1)), pL2J[1]))

    LL13 = np.vstack((LL1, LL3))
    LL23 = np.vstack((LL2, -LL3))

    # precondizionatore di B_y
    # estrazione dei blocchi di By
    H_list = [
        By[:2*n, :n+1],
        By[:2*n, n+1:2*n+1],
        By[:2*n, 2*n+1:3*n+2],
        By[2*n:4*n-1, n+1:2*n+1],
        By[6*n-1:8*n-2, 2*n+1:3*n+2]
    ]

    # calcolo dei precondizionatori di By
    pH123 = precond_circ_inplace([H_list[0], H_list[1], H_list[2]], (2*n, n), n)
    pH4J = precond_circ_inplace([H_list[3], H_list[4]], (2*n-2, n-1), n-1)

    HH1 = np.block([
        [pH123[0], pH123[1]],
        [np.zeros((2*n-1, n+1)), pH4J[0]],
        [pH123[2], pH123[1]]
    ])

    HH2 = np.block([
        [pH123[2], np.zeros((2*n, n))],
        [np.zeros((2*n-1, n+1)), np.zeros((2*n-1, n))],
        [pH123[0], np.zeros((2*n, n))]
    ])

    HH3 = np.hstack((pH4J[1], np.zeros((2*n-1, n))))

    HH13 = np.vstack((HH1, np.zeros((2*n-1, 2*n+1))))
    HH23 = np.vstack((HH2, HH3))

    # assemblaggio finale per B
    megaPrec_L, megaPrec_H = [], []
    for i in range(n):
        if i == 0:
            megaPrec_L = np.hstack((LL13, LL23))
            megaPrec_H = np.hstack((HH13, HH23))

        else:
            # creazione degli zeri
            rowZero_L = np.zeros(LL13.shape)
            rowZero_H = np.zeros(HH13.shape)
            for _ in range(1,i):
                rowZero_L = np.hstack((rowZero_L, np.zeros(LL13.shape)))
                rowZero_H = np.hstack((rowZero_H, np.zeros(HH13.shape)))
            rowZero_L = np.hstack((rowZero_L, LL13))
            rowZero_H = np.hstack((rowZero_H, HH13))

            colZero_L = np.zeros(LL23.shape)
            colZero_H = np.zeros(HH23.shape)
            for _ in range(1,i):
                colZero_L = np.vstack((colZero_L, np.zeros(LL23.shape)))
                colZero_H = np.vstack((colZero_H, np.zeros(HH23.shape)))

            megaPrec_L = np.block([
                [megaPrec_L, colZero_L],
                [rowZero_L, LL23]
            ])
            megaPrec_H = np.block([
                [megaPrec_H, colZero_H],
                [rowZero_H, HH23]
            ])

    megaPrec_L = megaPrec_L[:-(2*n-1), :-n]
    megaPrec_H = megaPrec_H[:-(2*n-1), :-n]

    return np.hstack((megaPrec_L.T, megaPrec_H.T))



# Precondizionatori
def apply_block_fourier_ext(A:NDArray, n:int, s:int, t:int) -> NDArray:
    """
    Applies (F_n ⊗ I_s)^* A (F_n ⊗ I_t)
    INPUT:
    - A:    the rectangular matrix
    - n:    eta
    - s,t:  s,t
    """
    assert A.shape == (n*s, n*t), f"Matrix A must be of shape ({n*s}, {n*t})"

    A_tensor = A.reshape(n, s, n, t)
    B_tensor = fft(ifft(A_tensor, axis=0), axis=2)
    return B_tensor.reshape(n*s, n*t)

def apply_block_fourier_in(A:NDArray, n:int, s:int, t:int) -> NDArray:
    """
    Applies (F_n ⊗ I_s) A (F_n ⊗ I_t)^*
    INPUT:
    - A:    the rectangular matrix
    - n:    eta
    - s,t:  s,t
    """
    assert A.shape == (n*s, n*t), f"Matrix A must be of shape ({n*s}, {n*t})"

    A_tensor = A.reshape(n, s, n, t)
    B_tensor = ifft(fft(A_tensor, axis=0), axis=2)
    return B_tensor.reshape(n*s, n*t)

def block_diagonal(A:NDArray, n:int, s:int, t:int) -> NDArray:
    """
    Extracts the block diagonal part of matrix A consisting of n blocks,
    each of size s x t. All off-diagonal blocks are set to zero.

    INPUT:
    - A:    the input matrix.
    - n:    eta
    - s,t:  s,t
    """
    # Validate input dimensions
    if A.shape != (n*s, n*t):
        raise ValueError(f"Input matrix A must have shape ({n*s}, {n*t}), but has shape {A.shape}.")

    A_tensor = A.reshape(n, s, n, t)                    # Reshape A to a 4D tensor: (n, s, n, t)
    mask = np.eye(n, dtype=bool).reshape(n, 1, n, 1)    # Create a mask for the block diagonal
    B_tensor = A_tensor * mask                          # Apply the mask: retain diagonal blocks, set others to zero

    # Reshape back to the original 2D matrix shape
    return B_tensor.reshape(n*s, n*t)

def apply_block_tau_ext(A:NDArray, n:int, s:int, t:int) -> NDArray:
    """
    Applies (T_n ⊗ I_s)^* A (T_n ⊗ I_t)
    INPUT:
    - A:    the rectangular matrix
    - n:    eta
    - s,t:  s,t
    """
    assert A.shape == (n*s, n*t), f"Matrix A must be of shape ({n*s}, {n*t})"

    first_col = A[:,0]
    last_row = A[-1,:]
    B_tensor = hankel(first_col, last_row)
    return (A - B_tensor)

def apply_block_tau_in(A:NDArray, n:int, s:int, t:int) -> NDArray:
    """
    Applies (T_n ⊗ I_s) A (T_n ⊗ I_t)^*
    INPUT:
    - A:    the rectangular matrix
    - n:    eta
    - s,t:  s,t
    """
    assert A.shape == (n*s, n*t), f"Matrix A must be of shape ({n*s}, {n*t})"

    first_col = A[:,0]
    last_row = A[-1,:]
    B_tensor = hankel(first_col, last_row)
    return (A - B_tensor)


def precond_circ_inplace(L: list[NDArray], n_diag:list[int], n_cut:int) -> list[NDArray]:
    '''Returns the list of block preconditioner from a list of block matrices.'''
    ans = []
    for i in L:
        if i.shape == n_diag:
            ans.append(i)

        elif i.shape[0] == 0:
            continue

        else:
            # c = np.real(apply_block_fourier_ext(
            #             block_diagonal(
            #                 apply_block_fourier_in(i[:n_diag[0],:n_diag[1]], n_cut, 2, 1),
            #                 n_cut, 2, 1),
            #             n_cut, 2, 1))
            
            c = np.real(block_diagonal(
                            apply_block_tau_in(i[:n_diag[0],:n_diag[1]], n_cut, 2, 1),
                            n_cut, 2, 1))

            if c.shape[0] != i.shape[0] and c.shape[1] != i.shape[1]:
                c = np.hstack((c, i[:c.shape[0], c.shape[1]:]))
                c = np.vstack((c, i[c.shape[0]:, :]))

            elif i.shape[0] > i.shape[1]:
                c = np.hstack((c, i[:, c.shape[1]:]))

            elif i.shape[0] < i.shape[1]:
                c = np.vstack((c, i[c.shape[0]:, :]))

            ans.append(c)

    return ans

def precond_tau_inplace(L: list[NDArray], n_diag:list[int], n_cut:int) -> list[NDArray]:
    '''Returns the list of block preconditioner from a list of block matrices.'''
    ans = []
    for i in L:
        if i.shape == n_diag:
            ans.append(i)

        elif i.shape[0] == 0:
            continue

        else:
            c = np.real(apply_block_tau_ext(
                        block_diagonal(
                            apply_block_tau_in(i[:n_diag[0],:n_diag[1]], n_cut, 2, 1),
                            n_cut, 2, 1),
                        n_cut, 2, 1))

            if c.shape[0] != i.shape[0] and c.shape[1] != i.shape[1]:
                c = np.hstack((c, i[:c.shape[0], c.shape[1]:]))
                c = np.vstack((c, i[c.shape[0]:, :]))

            elif i.shape[0] > i.shape[1]:
                c = np.hstack((c, i[:, c.shape[1]:]))

            elif i.shape[0] < i.shape[1]:
                c = np.vstack((c, i[c.shape[0]:, :]))

            ans.append(c)

    return ans

def precond_tau_rect(L: list[NDArray], n_diag:list[int], n_cut:int) -> list[NDArray]:
    '''Returns the list of block preconditioner from a list of block matrices.'''
    ans = []
    for i in L:
        if i.shape == n_diag:
            ans.append(i)

        elif i.shape[0] == 0:
            continue

        else:
            c = np.real(block_diagonal(
                            apply_block_tau_in(i[:(2*n_cut-2),:(2*n_cut-2)], n_cut-1, 2, 2),
                        n_cut-1, 2, 2))

            if c.shape[0] != i.shape[0] and c.shape[1] != i.shape[1]:
                c = np.hstack((c, np.zeros((c.shape[0], i.shape[1]-c.shape[1]))))
                c = np.vstack((c, np.zeros((i.shape[0]-c.shape[0], c.shape[1]))))

            elif i.shape[0] > i.shape[1]:
                c = np.vstack((c, np.zeros((i.shape[0]-c.shape[0], c.shape[1]))))

            elif i.shape[0] < i.shape[1]:
                c = np.hstack((c, np.zeros((c.shape[0], i.shape[1]-c.shape[1]))))

            ans.append(c)

    return ans


# Analisi spettrale
def SuperSymbol(n:int, case:int, gamma:int) ->list[float]:
    """Assembles the super symbol (aka the symbol of the full matrix) and
    evaluates its eigenvalues"""

    fullEigen = []
    for i in np.linspace(-np.pi, np.pi, n):
        for j in np.linspace(-np.pi, np.pi, n):

            sym_Bx = AssembleSymbolBx(i,j,n)
            sym_By = AssembleSymbolBy(i,j,n)

            for x in np.linspace(0,1,n):
                for y in np.linspace(0,1,n):
                    print(f"i = {i}\tj = {j}\tx = {x}\ty = {y}")
                    sym_Ax = AssembleSymbol_A(x,y,i,j,case,gamma)

                    # assembling the symbol
                    sym_A = np.kron(np.eye(2), sym_Ax)
                    sym_B = np.hstack((sym_Bx.T, sym_By.T))
                    super_sym  = np.block([
                        [sym_A, sym_B.T],
                        [sym_B, np.zeros((sym_B.shape[0], sym_B.shape[0]))]
                    ])

                    eig_G, _ = eigh(super_sym)
                    fullEigen.extend(eig_G)

    return np.sort(fullEigen, kind='heapsort')

def Symbol_Ax(n:int, case:int, gamma:int) ->list[float]:
    """Assembles the symbol of the matrix A_x and evaluates its eigenvalues"""

    fullEigen = []
    for i in np.linspace(-np.pi, np.pi, n):
        for j in np.linspace(-np.pi, np.pi, n):
            for x in np.linspace(0,1,n):
                for y in np.linspace(0,1,n):
                    print(f"i = {i}\tj = {j}\tx = {x}\ty = {y}")
                    eig_G, _ = eigh(AssembleSymbol_A(x,y,i,j,case,gamma))
                    fullEigen.extend(eig_G)

    return np.sort(fullEigen, kind='heapsort')

def AssembleSymbol_A(x:float, y:float, theta1:float, theta2:float, case:int, gamma:int):
    """Function that builds the symbol of the matrix sequence"""
    from cmath import exp
    G = np.zeros((8,8), dtype=np.complex128)

    match case:
        case 1:                     # constant
            mu = lambda x,y : 1 + 0*x + 0*y

        case 2:                     # smooth
            mu = lambda x,y : x*y + exp(x+y)

        case 3:                     # step function
            mu = lambda x,y : (gamma*(x<=.5 and y<=.5) + (1+x+y)*(x>.5 or y>.5))

        case _:
            raise ValueError("Invalid case value")
    
    h3 = 1/6 * (1+exp(complex(0,theta1)))*(1+exp(complex(0,theta2)))
    h3_conj = h3.conjugate()

    G[0,0] = 16/3
    G[0,2] = -4/3
    G[0,3] = -4/3 * exp(complex(0,theta2))
    G[0,6] = -4/3 * exp(complex(0,theta1))
    G[0,7] = -4/3 * exp(complex(0,theta2)+complex(0,theta1))

    G[1,1] = 16/3
    G[1,2] = -4/3
    G[1,3] = -4/3
    G[1,6] = -4/3 * exp(complex(0,theta1))
    G[1,7] = -4/3 * exp(complex(0,theta1))

    G[2,0] = -4/3
    G[2,1] = -4/3
    G[2,2] = 4
    G[2,4] = -4/3
    G[2,5] = -4/3 * exp(complex(0,theta2))
    G[2,7] = 2*h3

    G[3,0] = -4/3 * exp(complex(0,-theta2))
    G[3,1] = -4/3
    G[3,3] = 16/3
    G[3,4] = -4/3
    G[3,5] = -4/3

    G[4,2] = -4/3
    G[4,3] = -4/3
    G[4,4] = 16/3
    G[4,6] = -4/3
    G[4,7] = -4/3

    G[5,2] = -4/3 * exp(complex(0,-theta2))
    G[5,3] = -4/3
    G[5,5] = 16/3
    G[5,6] = -4/3 * exp(complex(0,-theta2))
    G[5,7] = -4/3

    G[6,0] = -4/3 * exp(complex(0,-theta1))
    G[6,1] = -4/3 * exp(complex(0,-theta1))
    G[6,4] = -4/3
    G[6,5] = -4/3 * exp(complex(0,theta2))
    G[6,6] = 16/3

    G[7,0] = -4/3 * exp(-(complex(0,theta2)+complex(0,theta1)))
    G[7,1] = -4/3 * exp(complex(0,-theta1))
    G[7,2] = 2*h3_conj
    G[7,4] = -4/3
    G[7,5] = -4/3
    G[7,7] = 4

    return mu(x,y)*G

def AssembleSymbolBx(theta1:float, theta2:float, n:int):
    """Function that builds the symbol of the Bx matrix.
    INPUT:
    - a: left side of the domain
    - b: right side of the domain
    - theta1: first variable for the evaluation of the symbol
    - theta2: second variable for the evaluation of the symbol
    - n: cells number used in the discretization
    """
    from cmath import exp
    G_const = np.zeros((8,2), dtype=np.complex128)
    G_mt1 = np.zeros((8,2), dtype=np.complex128)
    G_mt2 = np.zeros((8,2), dtype=np.complex128)
    G_mt1t2 = np.zeros((8,2), dtype=np.complex128)

    unsesto, undodicesimo = 1/6, 1/12

    G_const[0,0] = -unsesto
    G_const[1,0] = -undodicesimo
    G_const[4,0] = -undodicesimo
    G_const[0,1] = unsesto
    G_const[1,1] = unsesto
    G_const[4,1] = -unsesto
    G_const[5,1] = -unsesto
    G_const[6,1] = -unsesto

    G_mt1[0,0] = -undodicesimo
    G_mt1[1,0] = -unsesto
    G_mt1[3,0] = -unsesto
    G_mt1[5,0] = -undodicesimo

    G_mt2[0,0] = undodicesimo 
    G_mt2[4,0] = unsesto
    G_mt2[5,0] = undodicesimo
    G_mt2[6,1] = unsesto

    G_mt1t2[1,0] = undodicesimo
    G_mt1t2[3,0] = unsesto
    G_mt1t2[4,0] = undodicesimo
    G_mt1t2[5,0] = unsesto

    G = G_const + G_mt1*exp(complex(0, -theta1)) + G_mt2*exp(complex(0, -theta2)) + G_mt1t2*exp(complex(0, -(theta1+theta2)))
    return G

def AssembleSymbolBy(theta1:float, theta2:float, n:int):
    """Function that builds the symbol of the Bx matrix.
    INPUT:
    - a: left side of the domain
    - b: right side of the domain
    - theta1: first variable for the evaluation of the symbol
    - theta2: second variable for the evaluation of the symbol
    - n: cells number used in the discretization
    """
    from cmath import exp
    G_const = np.zeros((8,2), dtype=np.complex128)
    G_mt1 = np.zeros((8,2), dtype=np.complex128)
    G_mt2 = np.zeros((8,2), dtype=np.complex128)
    G_mt1t2 = np.zeros((8,2), dtype=np.complex128)

    unsesto, undodicesimo = 1/6, 1/12

    G_const[0,0] = -unsesto
    G_const[1,0] = -undodicesimo
    G_const[4,0] = -undodicesimo
    G_const[0,1] = unsesto
    G_const[1,1] = -unsesto
    G_const[3,1] = -unsesto
    G_const[4,1] = unsesto
    G_const[5,1] = -unsesto

    G_mt1[0,0] = undodicesimo
    G_mt1[1,0] = unsesto
    G_mt1[5,0] = undodicesimo
    G_mt1[3,1] = unsesto

    G_mt2[0,0] = -undodicesimo 
    G_mt2[4,0] = -unsesto
    G_mt2[5,0] = -undodicesimo
    G_mt2[6,0] = -unsesto

    G_mt1t2[1,0] = undodicesimo
    G_mt1t2[4,0] = undodicesimo
    G_mt1t2[5,0] = unsesto
    G_mt1t2[6,0] = unsesto

    G = G_const + G_mt1*exp(complex(0, -theta1)) + G_mt2*exp(complex(0, -theta2)) + G_mt1t2*exp(complex(0, -(theta1+theta2)))
    return G


def torchBlock(submatrices):
    rows = [np.cat(row, dim=1) for row in submatrices]
    return np.cat(rows, dim=0)