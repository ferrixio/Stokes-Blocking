# StokesMatrix is a script to build the stiffness matrix of the Stokes problem,
# according to the paper that Serra gave to us

# HOW TO USE IT:
# - Run the script (it is better if you use python 3.11)
# - Each cell is filled with a string containing a coefficient and a sum of mu_i,
#   writing only their INDEXES according to the enumeration found in the paper!!!
# - Import the data file in excel (or similar programs)

import numpy as np

class Bx_elements:
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
        return f"({self.i},{self.j}) -> entry: {self.mu} -> stiffIndex {self.stiffIndex}"
    

# generator functions
def ZeroColumnBx(n:int) -> tuple[list[Bx_elements], list[Bx_elements]]:
    """Returns the two lists of StiffElement to build the Bx matrix - Zero column"""
    even_rows, odd_rows = [], []

    # even part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1 + 2*k2
            # print(f"zero even cols k1={k1} k2={k2}\ti={i}")
            even_rows.extend([
                Bx_elements(i, (2*n+1)*k1+k2, f"-h/6"),
                Bx_elements(i, (2*n+1)*k1+k2+1, f"-h/12"),
                Bx_elements(i, (2*n+1)*k1+1+n+k2, f"h/6"),
                Bx_elements(i, (2*n+1)*k1+1+2*n+k2, f"h/12")]
            )

    # odd part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1+1+2*k2
            # print(f"zero odd cols k1={k1} k2={k2}\ti={i}")
            odd_rows.extend([
                Bx_elements(i, (2*n+1)*k1+k2, f"-h/12"),
                Bx_elements(i, (2*n+1)*k1+1+k2, f"-h/6"),
                Bx_elements(i, (2*n+1)*k1+1+n+k2, f"h/6"),
                Bx_elements(i, (2*n+1)*k1+2+2*n+k2, f"h/12")]
            )

    return even_rows, odd_rows

def FirstColumnBx(n:int) -> tuple[list[Bx_elements], list[Bx_elements]]:
    """Returns the two lists of StiffElement to build the Bx matrix - First + (n-1) columns"""
    odd_rows = []

    # odd part
    for k1 in range(n):
        for k2 in range(n-1):
            i = 2*n+1 + (8*n-2)*k1 + 2*k2
            # print(f"First odd cols k1={k1} k2={k2}\ti={i}")
            odd_rows.extend([
                Bx_elements(i, (2*n+1)*k1+1+k2, f"-h/6"),
                Bx_elements(i, (2*n+1)*k1+(2*n+2)+k2, f"h/6"),
            ])

    return odd_rows

def SecondColumnBx(n:int) -> tuple[list[Bx_elements], list[Bx_elements]]:
    """Returns the two lists of StiffElement to build the Bx matrix - Second + (n-1) columns"""
    even_rows, odd_rows = [], []

    # even part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1 +(4*n-1)+ 2*k2
            # print(f"Second even cols k1={k1} k2={k2}\ti={i}")
            even_rows.extend([
                Bx_elements(i, (2*n+1)*k1+k2, f"-h/12"),
                Bx_elements(i, (2*n+1)*k1+k2+n+1, f"-h/6"),
                Bx_elements(i, (2*n+1)*k1+(2*n+1)+k2, f"h/6"),
                Bx_elements(i, (2*n+1)*k1+(2*n+2)+k2, f"h/12"),
            ])

    # odd part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1 + 4*n + 2*k2
            # print(f"Second odd cols k1={k1} k2={k2}\ti={i}")
            odd_rows.extend([
                Bx_elements(i, (2*n+1)*k1+1+k2, f"-h/12"),
                Bx_elements(i, (2*n+1)*k1+n+1+k2, f"-h/6"),
                Bx_elements(i, (2*n+1)*k1+(2*n+1)+k2, f"h/12"),
                Bx_elements(i, (2*n+1)*k1+(2*n+2)+k2, f"h/6"),
            ])

    return even_rows, odd_rows

def FirstColumnn1Bx(n:int) -> tuple[list[Bx_elements], list[Bx_elements]]:
    """Returns the two lists of StiffElement to build the Bx matrix - First + (n-1) columns"""
    even_rows = []

    # even rows
    for k1 in range(n-1):
        for k2 in range(n):
            i = (8*n-2)*k1 + 6*n-1 + 2*k2
            # print(f"FirstColumnn1 even cols k1={k1} k2={k2}\t8/3 * ({3}+{4*n})\ti={i}")

            even_rows.extend([
                Bx_elements(i, (2*n+1)*k1+n+1+k2, f"-h/6"),
                Bx_elements(i, (2*n+1)*k1+(3*n+2)+k2, f"h/6"),
            ])

    return even_rows


def AssembleBx(n:int, *arg) -> tuple[np.array, list]:
    """Function that build the Bx matrix"""
    Nv = 8*(n**2)-4*n+1     # size of the matrix

    # Determine the maximum j index from all Bx_elements to set Nq appropriately
    max_j = max(stiff.j for item in arg for stiff in item)
    Nq = max_j + 1
    Bx = np.full((Nv, Nq), None, dtype=object)

    res = []
    for item in arg:
        for stiff in item:
            # print(stiff)
            # if the position is empty, fill it
            if Bx[stiff.i, stiff.j] is None:
                Bx[stiff.i, stiff.j] = stiff.mu

            # else there is a residual and append to an external list
            else:
                res.append([str(stiff), f"item {arg.index(item)}, stiff index {item.index(stiff)}"])
                # Optionally, you can handle duplicates by concatenating the strings or other logic
                # For example:
                # A[stiff.i, stiff.j] += f" + {stiff.mu}"

    return Bx, res

class By_elements:
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
def ZeroColumnBy(n:int) -> tuple[list[By_elements], list[By_elements]]:
    """Returns the two lists of StiffElement to build the By matrix - Zero column"""
    even_rows, odd_rows = [], []

    # even part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1 + 2*k2
            # print(f"zero even cols k1={k1} k2={k2}\ti={i}")
            even_rows.extend([
                By_elements(i, (2*n+1)*k1+k2, f"-h/6"),
                By_elements(i, (2*n+1)*k1+k2+1, f"h/12"),
                By_elements(i, (2*n+1)*k1+1+n+k2, f"h/6"),
                By_elements(i, (2*n+1)*k1+1+2*n+k2, f"-h/12")]
            )

    # odd part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1 + 1 + 2*k2
            # print(f"zero odd cols k1={k1} k2={k2}\ti={i}")
            odd_rows.extend([
                By_elements(i, (2*n+1)*k1+k2, f"-h/12"),
                By_elements(i, (2*n+1)*k1+k2+1, f"h/6"),
                By_elements(i, (2*n+1)*k1+1+n+k2, f"-h/6"),
                By_elements(i, (2*n+1)*k1+2+2*n+k2, f"h/12")]
            )

    return even_rows, odd_rows

def FirstColumnBy(n:int) -> tuple[list[By_elements], list[By_elements]]:
    """Returns the two lists of StiffElement to build the By matrix - First + (n-1) columns"""
    odd_rows = []

    # odd part
    for k1 in range(n):
        for k2 in range(n-1):
            i = 2*n+1 + (8*n-2)*k1 + 2*k2
            # print(f"First odd cols k1={k1} k2={k2}\ti={i}")
            odd_rows.extend([
                By_elements(i, (2*n+1)*k1+n+1+k2, f"-h/6"),
                By_elements(i, (2*n+1)*k1+n+2+k2, f"h/6"),
            ])

    # NOTE: the even part is zero

    return odd_rows

def SecondColumnBy(n:int) -> tuple[list[By_elements], list[By_elements]]:
    """Returns the two lists of StiffElement to build the By matrix - Second + (n-1) columns"""
    even_rows, odd_rows = [], []

    # even part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1 +(4*n-1)+ 2*k2
            # print(f"Second even cols k1={k1} k2={k2}\ti={i}")
            even_rows.extend([
                By_elements(i, (2*n+1)*k1+k2, f"-h/12"),
                By_elements(i, (2*n+1)*k1+k2+n+1, f"h/6"),
                By_elements(i, (2*n+1)*k1+(2*n+1)+k2, f"-h/6"),
                By_elements(i, (2*n+1)*k1+(2*n+2)+k2, f"h/12"),
            ])

    # odd part
    for k1 in range(n):
        for k2 in range(n):
            i = (8*n-2)*k1 + 4*n + 2*k2
            # print(f"Second odd cols k1={k1} k2={k2}\ti={i}")
            odd_rows.extend([
                By_elements(i, (2*n+1)*k1+1+k2, f"h/12"),
                By_elements(i, (2*n+1)*k1+n+1+k2, f"-h/6"),
                By_elements(i, (2*n+1)*k1+(2*n+1)+k2, f"-h/12"),
                By_elements(i, (2*n+1)*k1+(2*n+2)+k2, f"h/6"),
            ])

    return even_rows, odd_rows

def FirstColumnn1By(n:int) -> tuple[list[By_elements], list[By_elements]]:
    """Returns the two lists of StiffElement to build the By matrix - First + (n-1) columns"""
    even_rows = []

    # even rows
    for k1 in range(n-1):
        for k2 in range(n):
            i = (8*n-2)*k1 + 6*n-1 + 2*k2
            # print(f"FirstColumnn1 even cols k1={k1} k2={k2}\t8/3 * ({3}+{4*n})\ti={i}")
            even_rows.extend([
                By_elements(i, (2*n+1)*k1+(2*n+1)+k2, f"-h/6"),
                By_elements(i, (2*n+1)*k1+(2*n+2)+k2, f"h/6"),
            ])

    # NOTE: the odd part is zero

    return even_rows


def AssembleBy(n:int, *arg) -> tuple[np.array, list]:
    """Function that build the By matrix"""
    Nv = 8*(n**2)-4*n+1     # size of the matrix

    # Determine the maximum j index from all By_elements to set Nq appropriately
    max_j = max(stiff.j for item in arg for stiff in item)
    Nq = max_j + 1
    By = np.full((Nv, Nq), None, dtype=object)

    res = []
    for item in arg:
        for stiff in item:
            # print(stiff)
            # if the position is empty, fill it
            if By[stiff.i, stiff.j] is None:
                By[stiff.i, stiff.j] = stiff.mu

            # else there is a residual and append to an external list
            else:
                res.append([str(stiff), f"item {arg.index(item)}, stiff index {item.index(stiff)}"])
                # Optionally, you can handle duplicates by concatenating the strings or other logic
                # For example:
                # A[stiff.i, stiff.j] += f" + {stiff.mu}"

    return By, res


## MAIN ##
if __name__ == "__main__":

    # partition of the domain
    n = 4

    # generating the elements
    zeroEBx, zeroOBx = ZeroColumnBx(n)
    secondEBx, secondOBx = SecondColumnBx(n)

    zeroEBy, zeroOBy = ZeroColumnBy(n)
    secondEBy, secondOBy = SecondColumnBy(n)

    # assembling the big matrices and get the residuals
    GlobalMatrixBx, resBx = AssembleBx(n,
                                       zeroEBx, zeroOBx,
                                       FirstColumnBx(n),
                                       secondEBx, secondOBx,
                                       FirstColumnn1Bx(n))
    
    GlobalMatrixBy, resBy = AssembleBy(n,
                                       zeroEBy, zeroOBy,
                                       FirstColumnBy(n),
                                       secondEBy, secondOBy,
                                       FirstColumnn1By(n))

    # print residuals
    print(f"Residuals found in Bx: {len(resBx)}")
    print(*resBx, sep='\n')
    print(f"Residuals found in By: {len(resBy)}")
    print(*resBy, sep='\n')

    # save matrices on .csv files
    np.savetxt("FullGlobalBx.csv", GlobalMatrixBx, delimiter=';', fmt="'%s'")
    np.savetxt("FullGlobalBy.csv", GlobalMatrixBy, delimiter=';', fmt="'%s'")