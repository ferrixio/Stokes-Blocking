from StokesBuilder import *

def SymbolEigenvalues(n: int) -> list[float]:
    """Evaluate the eigenvalues of the symbol in a (uniform) sampling of size n"""

    fullSVDBx, fullSVDBy = [], []
    for i in np.linspace(-np.pi, np.pi, n):
        for j in np.linspace(-np.pi, np.pi, n):
            fullSVDBx.extend(np.linalg.svd(AssembleSymbolBx(i,j), compute_uv=False))
            fullSVDBy.extend(np.linalg.svd(AssembleSymbolBy(i,j), compute_uv=False))

    return np.sort(fullSVDBx), np.sort(fullSVDBy)



## MAIN ##
if __name__ == "__main__":

    # partition of the domain
    n = 16

    # assemble B_x
    zeroEBx, zeroOBx = ZeroColumnBx(n)
    secondEBx, secondOBx = SecondColumnBx(n)
    Bx, resBx = AssembleB_block(n, zeroEBx, zeroOBx, FirstColumnBx(n), secondEBx, secondOBx, FirstColumnn1Bx(n))

    # assemble B_y
    zeroEBy, zeroOBy = ZeroColumnBy(n)
    secondEBy, secondOBy = SecondColumnBy(n)
    By, resBy = AssembleB_block(n, zeroEBy, zeroOBy, FirstColumnBy(n), secondEBy, secondOBy, FirstColumnn1By(n))

    sing1, sing2 = SymbolEigenvalues(1000)
    xx = np.linspace(0,1, len(sing1))

    # compute svd
    svd_x = np.linalg.svd(Bx, compute_uv=False)
    svd_y = np.linalg.svd(By, compute_uv=False)
    yy = np.linspace(0,1, len(svd_x))

    plt.figure(1)
    plt.plot(yy, np.sort(svd_x), '.', xx, sing1)
    plt.legend([r'SVD $B_{x,n}$', 'Symbol'])
    
    plt.figure(2)
    plt.plot(yy, np.sort(svd_y), '.', xx, sing2)
    plt.legend([r'SVD $B_{y,n}$', 'Symbol'])

    plt.show()