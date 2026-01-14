from StokesBuilder import *

def Permute(A:NDArray, n:int) -> NDArray:
    '''Applies the permutation to the matrix A.

    Memo:
    - AP = column permutation
    - PA = row permutation
    '''
    
    return A


def IsolateBlockBy(By:NDArray, n:int) -> NDArray:
    '''Returns the generator block of the matrix By.

    INPUT:
    - By = the matrix By
    - n = the basic division of the domain

    OUTPUT:
    - the generator of By
    '''
    # COMMENT: data la struttura di By, il blocco generatore è 4x4 della forma:
    #   H1 H2 H3 0      H1.shape = 2n x n+1     K.shape = 2n-1 x n+1
    #   0  H4 0  0      H2.shape = 2n x n       (quarta col di 0).col = n
    #   H3 H2 H1 0      H3.shape = 2n x n+1
    #   0  0  K  0      H4.shape = 2n-1 x n
    # Quindi la dimensione vera è:
    #   - righe = 2n + 2n-1 + 2n + 2n-1 = 8n-2 = 2(4n-1)
    #   - colonne = n+1 + n + n+1 + n = 4n+2 = 2(2n+1)

    return By[:8*n-2, :4*n+2]


def GetHalves(By:NDArray, n:int) -> tuple[NDArray]:
    '''
    Returns the two halves of the generator of the matrix By.
    
    :param By: The generator of By
    :type By: NDArray
    :param n: The basic division of the domain
    :type n: int
    :return: The two halves as NDArray
    :rtype: tuple[NDArray]
    '''
    # COMMENT: data la struttura del generatore, le due metà saranno:
    #       H1 H2       H3 0
    #       0  H4       0  0
    #       H3 H2       H1 0
    #       0  0        K  0

    return By[:, :2*n+1], By[:, 2*n+2:]


def ShifterKron(half1:NDArray, half2:NDArray, n:int) -> NDArray:
    '''Applies the shifted tensor product.
    It performs eye(n) /tensor half1 + eye(n,k=1) /tensor half2.'''

    return np.kron(np.eye(n), half1) + np.kron(np.eye(n, k=1), half2)