'''
Group 2: testing the adeherence of the eigenvalues of the full matrix 
against the spectral symbol.

mu(x,y) = xy + exp(x+y)
'''

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.StokesBuilder import *

if __name__ == "__main__":

    n = 24                          # order of division
    coeffMu = CreateMu(n, 2, 0)     # viscosity coefficients


    # Assembly of matrix A
    A_xx, res = AssembleStiffness(n, *ZeroColumn(n, coeffMu), *SecondColumn(n, coeffMu), *ThirdColumn(n, coeffMu),
                                          *FourthColumn(n, coeffMu), *LastColumn(n, coeffMu), *Centers(n, coeffMu))

    BIG_A = np.kron(np.eye(2), A_xx)

    # Assembly of matrix Bx
    zeroEBx, zeroOBx = ZeroColumnBx(n)
    secondEBx, secondOBx = SecondColumnBx(n)
    Bx, resBx = AssembleB_block(n, zeroEBx, zeroOBx, FirstColumnBx(n), secondEBx, secondOBx, FirstColumnn1Bx(n))

    # Assembly of matrix By
    zeroEBy, zeroOBy = ZeroColumnBy(n)
    secondEBy, secondOBy = SecondColumnBy(n)
    By, resBy = AssembleB_block(n, zeroEBy, zeroOBy, FirstColumnBy(n), secondEBy, secondOBy, FirstColumnn1By(n))

    BIG_B = np.hstack((Bx.T, By.T))

    # Final assembly of matrix M
    SUPER_A = np.block([
        [BIG_A, BIG_B.T],
        [BIG_B, np.zeros((BIG_B.shape[0], BIG_B.shape[0]))]
    ])

    # Symbol's sampling
    fullEigen = SuperSymbol(n, 2, 0)
    fullEigenAxx = Symbol_Ax(n, 2, 0)
    xxe = np.linspace(0,1,len(fullEigen))
    xxeAx = np.linspace(0,1,len(fullEigenAxx))

    sing, _ = eig(SUPER_A)
    xx = np.linspace(0,1,len(sing))

    singAxx, _ = eig(A_xx)
    yy = np.linspace(0,1,len(singAxx))

    plt.figure(1)
    plt.plot(xx, np.sort(sing, kind='heapsort'), '*', xxe, fullEigen)
    plt.legend([r'$\lambda_j(M_n)$', "$f(x,y,\\theta_1,\\theta_2)$"], fontsize=14)

    plt.figure(2)
    plt.plot(yy, np.sort(singAxx, kind='heapsort'), '*', xxe, fullEigen)
    plt.legend([r'$\lambda_j(A_{x,n})$', "$f(x,y,\\theta_1,\\theta_2)$"], fontsize=14)

    plt.show()