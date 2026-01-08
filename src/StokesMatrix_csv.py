# StokesMatrix is a script to build the stiffness matrix of the Stokes problem,
# according to the paper that Serra gave to us

# HOW TO USE IT:
# - Run the script (it is better if you use python 3.11)
# - Each cell is filled with a string containing a coefficient and a sum of mu_i,
#   writing only their INDEXES according to the enumeration found in the paper!!!
# - Import the data file in excel (or similar programs)

import numpy as np

class StiffElement:
    '''Object that stores informations to build the stiffness matrix
    
    ATTRIBUTES:
    - i [int]                    = row index in the big stiffness matrix
    - j [int]                    = column index in the big stiffness matrix
    - mu [str]                   = string to assign at the position (i,j)
    - stiffIndex [list[int,int]] = relative position during the construction of the elements
    '''
    def __init__(self, i:int, j:int, item:str='dummy', stiffIndex:list=[]) -> None:
        self.i = i
        self.j = j
        self.mu = item
        self.stiffIndex = stiffIndex

    def __str__(self) -> str:
        '''Magic method to print the element in a pretty way using print()'''
        return f"({self.i},{self.j}) -> mu: {self.mu} -> stiffIndex {self.stiffIndex}"
    

# generator functions
def ZeroColumn(n:int) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the global stiffness matrix
    at the end - Zero column
    """
    elem_even, elem_odd = [], []

    # even part
    for k1 in range(n):
        # print(f"Zero even cols {k1}")
        i = 2*k1
        elem_even.extend([
            StiffElement(i, i, f"8/3 * ({4*k1}+{1+4*k1})"),
            StiffElement(i, i+2*n, f"-2/3 * ({4*k1}+{1+4*k1})")
        ])
        if 0!=k1:
            elem_even.append(StiffElement(i, i+2*n-1, f"-4/3 * ({1+4*k1})"))

    # odd part
    for k1 in range(n):
        # print(f"Zero odd cols {k1}")
        i = 1+2*k1
        elem_odd.extend([
            StiffElement(i, i, f"8/3 * ({4*k1}+{2+4*k1})"),
            StiffElement(i, i+2*n-1, f"-2/3 * ({4*k1}+{2+4*k1})")
        ])
        if (n-1)!=k1:
            elem_odd.append(StiffElement(i, i+2*n, f"-4/3 * ({2+4*k1})"))

    return elem_even, elem_odd

def SecondColumn(n:int) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the global stiffness matrix
    at the end - Second + (n-1) columns"""
    even_rows, odd_rows = [], []

    # even part
    for k1 in range(n-1):
        for k2 in range(n):
            i = 4*n-1 + (8*n-2)*k1 + 2*k2
            # print(f"Second even cols k1={k1} k2={k2}\ti={i}")
            even_rows.extend([
                StiffElement(i, i-2*n+1, f"-2/3 * ({1+4*n*k1+4*k2}+{3+4*n*k1+4*k2})"),
                StiffElement(i, i, f"8/3 * ({1+4*n*k1+4*k2}+{3+4*n*k1+4*k2})"),
                StiffElement(i, i+2*n, f"-4/3 * ({3+4*n*k1+4*k2})")]
            )

            if 0!=k2:
                even_rows.extend([
                    StiffElement(i,i-2*n, f"-4/3 * ({1+4*n*k1+4*k2})"),
                    StiffElement(i,i+2*n-1, f"-2/3 * ({1+4*n*k1+4*k2}+{3+4*n*k1+4*k2})")]
                    )

    # odd part
    for k1 in range(n-1):
        for k2 in range(n):
            i = 4*n + (8*n-2)*k1 + 2*k2
            # print(f"Second odd cols k1={k1} k2={k2}\ti={i}")
            odd_rows.extend([
                StiffElement(i, i-2*n, f"-2/3 * ({2+4*n*k1+4*k2}+{3+4*n*k1+4*k2})"),
                StiffElement(i, i, f"8/3 * ({2+4*n*k1+4*k2}+{3+4*n*k1+4*k2})"),
                StiffElement(i, i+2*n-1, f"-4/3 * ({3+4*n*k1+4*k2})")]
            )

            if (n-1)!=k2:
                odd_rows.extend([
                    StiffElement(i,i-2*n+1, f"-4/3 * ({2+4*n*k1+4*k2})"),
                    StiffElement(i,i+2*n, f"-2/3 * ({2+4*n*k1+4*k2}+{3+4*n*k1+4*k2})")]
                )

    return even_rows, odd_rows

def Centers(n:int) -> tuple[list[StiffElement], list[StiffElement], list[StiffElement], list[StiffElement]]:
    """Returns the evaluations in the centers of cells"""
    bottom_row, middle_bottom, middle_top, top_row = [], [], [], []
    small = 8*n-2

    #bottom part
    for k1 in range(n):
        i = 2*n + small*k1
        base = 4*n*k1
        # print(f"Center bottom cols k1={k1}\t({base}+{1+base}+{2+base}+{3+base})\ti={i}")
        bottom_row.extend([
            StiffElement(i, i-2*n, f"-2/3 * ({base}+{1+base})"),
            StiffElement(i, i-2*n+1, f"-2/3 * ({base}+{2+base})"),
            StiffElement(i, i,f"({base}+{1+base}+{2+base}+{3+base})"),
            StiffElement(i, i+2*n-1, f"-2/3 * ({1+base}+{3+base})"),
            StiffElement(i, i+2*n, f"-2/3 * ({2+base}+{3+base})")]
        )

        if k1 != 0:
            bottom_row.append(
                StiffElement(i, i-4*n+2, f"1/6 * ({base}+{2+base})")
            )

        if k1 != (n-1):
            bottom_row.append(
                StiffElement(i, i+4*n, f"1/6 * ({2+base}+{3+base})")
            )

    # middle bottom part
    for k1 in range(n):
        for k2 in range(n-1):
            i = 2*n+1 + small*k1 + 2*k2
            base = 4*n*k1+4*k2
            # print(f"Center middle bottom cols k1={k1} k2={k2}\t8/3 * ({2+base}+{5+base})\ti={i}")
            middle_bottom.extend([
                StiffElement(i, i-2*n, f"-4/3 * ({2+base})"),
                StiffElement(i, i-2*n+1, f"-4/3 * ({5+base})"),
                StiffElement(i, i, f"8/3 * ({2+base}+{5+base})"),
                StiffElement(i, i+2*n-1, f"-4/3 * ({2+base})"),
                StiffElement(i, i+2*n, f"-4/3 * ({5+base})")]
            )

    # middle top part
    for k1 in range(n):
        for k2 in range(1,n-1):
            i = 2*n + small*k1 + 2*k2
            base = 4*n*k1+4*k2
            # print(f"Center middle top cols k1={k1} k2={k2}\t({base}+{1+base}+{2+base}+{3+base})\ti={i}")
            middle_top.extend([
                StiffElement(i, i-2*n, f"-2/3 * ({base}+{1+base})"),
                StiffElement(i, i-2*n+1, f"-2/3 * ({base}+{2+base})"),
                StiffElement(i, i, f"({base}+{1+base}+{2+base}+{3+base})"),
                StiffElement(i, i+2*n-1, f"-2/3 * ({1+base}+{3+base})"),
                StiffElement(i, i+2*n, f"-2/3 * ({2+base}+{3+base})")]
            )

            if k1 != 0:
                middle_top.extend([
                    StiffElement(i, i-4*n, f"1/6 * ({base}+{1+base})"),
                    StiffElement(i, i-4*n+2, f"1/6 * ({base}+{base+2})")]
                )

            if k1 != (n-1):
                middle_top.extend([
                    StiffElement(i, i+4*n-2, f"1/6 * ({1+base}+{3+base})"),
                    StiffElement(i, i+4*n, f"1/6 * ({2+base}+{3+base})")]
                )

    # top row part
    for k1 in range(n):
        i = 4*n-2 + small*k1
        base = 4*n*k1 + 4*(n-1)
        # print(f"Center top cols k1={k1}\t({base}+{1+base}+{2+base}+{3+base})\ti={i}")
        top_row.extend([
            StiffElement(i, i-2*n, f"-2/3 * ({base}+{1+base})"),
            StiffElement(i, i-2*n+1, f"-2/3 * ({base}+{2+base})"),
            StiffElement(i, i,f"({base}+{1+base}+{2+base}+{3+base})"),
            StiffElement(i, i+2*n-1, f"-2/3 * ({1+base}+{3+base})"),
            StiffElement(i, i+2*n, f"-2/3 * ({2+base}+{3+base})")]
        )

        if k1 != 0:
            top_row.append(
                StiffElement(i, i-4*n, f"1/6 * ({base}+{1+base})")
            )

        if k1 != (n-1):
            top_row.append(
                StiffElement(i, i+4*n-2, f"1/6 * ({1+base}+{3+base})")
            )

    return bottom_row, middle_bottom, middle_top, top_row

def ThirdColumn(n:int) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the global stiffness matrix
    at the end - Third + (n-1) columns"""
    even_rows, odd_rows = [], []
    small = 8*n-2

    # even rows
    for k1 in range(n-1):
        for k2 in range(n):
            i = 6*n-1 + small*k1 + 2*k2
            base = 4*n*k1+4*k2
            # print(f"Third even cols k1={k1} k2={k2}\t8/3 * ({3+base}+{4*n+base})\ti={i}")

            even_rows.extend([
                StiffElement(i, i-2*n, f"-4/3 * ({3+base})"),
                StiffElement(i, i-2*n+1, f"-4/3 * ({3+base})"),
                StiffElement(i, i, f"8/3 * ({3+base}+{4*n+base})"),
                StiffElement(i, i+2*n-1, f"-4/3 * ({4*n+base})"),
                StiffElement(i, i+2*n, f"-4/3 * ({4*n+base})")]
            )

    # odd rows
    for k1 in range(n-1):
        for k2 in range(n-1):
            i = 6*n + small*k1 + 2*k2
            base = 4*n*k1+4*k2
            # print(f"Third odd cols k1={k1} k2={k2}\t1/2 * ({2+base}+{3+base}+{5+base}+{7+base}+{4*n+base}+{4*n+2+base}+{4*n+4+base}+{4*n+5+base})\ti={i}")

            odd_rows.extend([
                StiffElement(i, i-4*n, f"1/6 * ({2+base}+{3+base})"),
                StiffElement(i, i-4*n+2, f"1/6 * ({5+base}+{7+base})"),
                StiffElement(i, i-2*n, f"-2/3 * ({2+base}+{3+base})"),
                StiffElement(i, i-2*n+1, f"-2/3 * ({5+base}+{7+base})"),
                StiffElement(i, i, f"1/2 * ({2+base}+{3+base}+{5+base}+{7+base}+{4*n+base}+{4*n+2+base}+{4*n+4+base}+{4*n+5+base})"),
                StiffElement(i, i+2*n-1, f"-2/3 * ({4*n+base}+{4*n+2+base})"),
                StiffElement(i, i+2*n, f"-2/3 * ({4*n+4+base}+{4*n+5+base})"),
                StiffElement(i, i+4*n-2, f"1/6 * ({4*n+base}+{4*n+2+base})"),
                StiffElement(i, i+4*n, f"1/6 * ({4*n+4+base}+{4*n+5+base})")]
            )

    return even_rows, odd_rows

def FourthColumn(n:int) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the global stiffness matrix
    at the end - Fourth + (n-1) columns"""
    even_rows, odd_rows = [], []
    small = 8*n-2

    # even part
    for k1 in range(1, n):
        for k2 in range(n):
            i = small*k1 + 2*k2
            base = 4*n*k1+4*k2
            # print(f"Fourth even cols k1={k1} k2={k2}\ti={i}")

            even_rows.extend([
                StiffElement(i, i-2*n+1, f"-4/3 * ({base})", stiffIndex=(k1,k2,1)),
                StiffElement(i, i, f"8/3 * ({base}+{1+base})", stiffIndex=(k1,k2,2)),
                StiffElement(i, i+2*n, f"-2/3 * ({base}+{1+base})", stiffIndex=(k1,k2,3))]
            )

            if k2 != 0:
                even_rows.extend([
                    StiffElement(i, i-2*n, f"-2/3 * ({base}+{1+base})", stiffIndex=(k1,k2,'d1')),
                    StiffElement(i, i+2*n-1, f"-4/3 * ({1+base})", stiffIndex=(k1,k2,'d2'))]
                )

    # odd part
    for k1 in range(1, n):
        for k2 in range(n):
            i = 1 + small*k1 + 2*k2
            base = 4*n*k1+4*k2
            # print(f"Fourth odd cols k1={k1} k2={k2}\ti={i}")

            odd_rows.extend([
                StiffElement(i, i-2*n, f"-4/3 * ({base})"),
                StiffElement(i, i, f"8/3 * ({base}+{2+base})"),
                StiffElement(i, i+2*n-1, f"-2/3 * ({base}+{2+base})")]
            )

            if k2 != (n-1):
                odd_rows.extend([
                    StiffElement(i, i-2*n+1, f"-2/3 * ({base}+{2+base})"),
                    StiffElement(i, i+2*n, f"-4/3 * ({2+base})")]
                )


    return even_rows, odd_rows

def LastColumn(n:int) -> tuple[list[StiffElement], list[StiffElement]]:
    """Returns the two lists of StiffElement to build the global stiffness matrix
    at the end - last column"""
    even_rows, odd_rows = [], []
    small = 8*(n**2)-6*n+1

    # even part
    for k1 in range(n):
        i = small + 2*k1
        base = 4*n*(n-1)+4*k1
        # print(f"Last even cols k1={k1}\ti={i}")
        
        even_rows.extend([
            StiffElement(i, i-2*n+1, f"-2/3 * ({1+base}+{3+base})"),
            StiffElement(i, i, f"8/3 * ({1+base}+{3+base})")]
        )

        if k1 != 0:
            even_rows.append(StiffElement(i, i-2*n, f"-4/3 * ({1+base})"))

    # odd part
    for k1 in range(n):
        i = small +2*k1+1
        base = 4*n*(n-1)+4*k1
        # print(f"Last odd cols k1={k1}\ti={i}")

        odd_rows.extend([
            StiffElement(i, i-2*n, f"-2/3 * ({2+base}+{3+base})"),
            StiffElement(i, i, f"8/3 * ({2+base}+{3+base})")]
        )

        if k1 != (n-1):
            odd_rows.append(StiffElement(i, i-2*n+1, f"-4/3 * ({2+base})"))


    return even_rows, odd_rows


def AssembleStiffness(n:int, *arg) -> np.array:
    """Function that build the big stiffness matrix from a list of Stiffelements"""
    Nv = 8*(n**2)-4*n+1     # size of the matrix
    A = np.empty((Nv, Nv), dtype=np.object_)

    res = []
    for item in arg:
        for stiff in item:
            # print(stiff)
            # if the position is empty, fill it
            if not A[stiff.i, stiff.j]:
                A[stiff.i, stiff.j] = stiff.mu

            # else there is a residual and append to an external list
            # TODO: why this happens and why only in "FourthColumn" method?
            else:
                res.append([stiff.__str__(), f"item {arg.index(item)}, stiff index {item.index(stiff)}"])
                # raise ValueError(f'Error in position {stiff.i},{stiff.j}')
                # A[stiff.i, stiff.j] += stiff.mu

    return A, res


## MAIN ##
if __name__ == "__main__":

    # partition of the domain
    n = 4

    # generating the elements
    zeroE, zeroO = ZeroColumn(n)
    secondE, secondO = SecondColumn(n)
    thirdE, thirdO = ThirdColumn(n)
    fourthE, fourthO = FourthColumn(n)
    lastE, lastO = LastColumn(n)
    center1, center2, center3, center4 = Centers(n)
    
    # assembling the big matrix and get the residuals
    GlobalMatrix, res = AssembleStiffness(n,
                                        zeroE, zeroO,
                                        secondE, secondO,
                                        thirdE, thirdO,
                                        fourthE, fourthO,
                                        lastE, lastO,
                                        center1, center2, center3, center4)
    
    # print residuals
    # print(f"Residuals found: {len(res)}")
    # print(*res, sep='\n')

    # save matrix on a .csv file
    np.savetxt("FullGlobal.csv", GlobalMatrix, delimiter=';', fmt="'%s'")
