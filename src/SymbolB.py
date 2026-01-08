from StokesBuilder import *

def SymbolEigenvalues(n: int) -> list[float]:
    """Evaluate the eigenvalues of the symbol in a (uniform) sampling of size n"""

    fullSVDBx, fullSVDBy = [], []
    for i in np.linspace(-np.pi, np.pi, n):
        for j in np.linspace(-np.pi, np.pi, n):
            fullSVDBx.extend(np.linalg.svd(AssembleSymbolBx(i,j,n=n), compute_uv=False))
            fullSVDBy.extend(np.linalg.svd(AssembleSymbolBy(i,j,n=n), compute_uv=False))

    return np.sort(fullSVDBx), np.sort(fullSVDBy)



## MAIN ##
if __name__ == "__main__":

    # partition of the domain
    n = 8

    # assemblaggio B_x
    # zeroEBx, zeroOBx = ZeroColumnBx(n)
    # secondEBx, secondOBx = SecondColumnBx(n)
    # Bx, resBx = AssembleB_block(n, zeroEBx, zeroOBx, FirstColumnBx(n), secondEBx, secondOBx, FirstColumnn1Bx(n))

    # assemblaggio B_y
    # zeroEBy, zeroOBy = ZeroColumnBy(n)
    # secondEBy, secondOBy = SecondColumnBy(n)
    # By, resBy = AssembleB_block(n, zeroEBy, zeroOBy, FirstColumnBy(n), secondEBy, secondOBy, FirstColumnn1By(n))

    sing1, sing2 = SymbolEigenvalues(n)
    xx = np.linspace(0,len(sing1), len(sing1))

    plt.semilogy(xx, sing1, '*', xx, sing2, 'x')
    plt.legend([r'svd sym($B_x$)', r'svd sym($B_y$)'])
    plt.show()
    