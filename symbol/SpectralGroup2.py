'''
COMMENT: 2nd group of tests

mu = xy + exp(x+y)

Tested the adeherence of the eigenvalues of the full matrix against the spectral symbol
'''

from src.StokesBuilder import *

if __name__ == "__main__":

    n = 16                          # ordine della divisione
    coeffMu = CreateMu(n, 2, 0)     # ottenimento dei coefficienti di mu


    # STEP: creazione matrice A
    # assemblaggio A_xx
    A_xx, res = AssembleStiffness(n, *ZeroColumn(n, coeffMu), *SecondColumn(n, coeffMu), *ThirdColumn(n, coeffMu),
                                          *FourthColumn(n, coeffMu), *LastColumn(n, coeffMu), *Centers(n, coeffMu))

    BIG_A = np.kron(np.eye(2), A_xx)    # assemblaggio A


    # STEP: creazione matrice B
    # assemblaggio B_x
    zeroEBx, zeroOBx = ZeroColumnBx(n)
    secondEBx, secondOBx = SecondColumnBx(n)
    Bx, resBx = AssembleB_block(n, zeroEBx, zeroOBx, FirstColumnBx(n), secondEBx, secondOBx, FirstColumnn1Bx(n))

    # assemblaggio B_y
    zeroEBy, zeroOBy = ZeroColumnBy(n)
    secondEBy, secondOBy = SecondColumnBy(n)
    By, resBy = AssembleB_block(n, zeroEBy, zeroOBy, FirstColumnBy(n), secondEBy, secondOBy, FirstColumnn1By(n))

    BIG_B = np.hstack((Bx.T, By.T))

    # assemblaggio matriciona
    SUPER_A = np.block([
        [BIG_A, BIG_B.T],
        [BIG_B, np.zeros((BIG_B.shape[0], BIG_B.shape[0]))]
    ])


    # STEP: ottenimento del simbolo
    fullEigen = SuperSymbol(16, 2, 0)    
    xxe = np.linspace(0,1,len(fullEigen))

    sing, _ = eig(SUPER_A)
    xx = np.linspace(0,1,len(sing))

    plt.figure(1)
    plt.plot(xx, np.sort(sing, kind='heapsort'), '*', xxe, fullEigen)
    plt.legend([r'$M_n$', "Symbol"], fontsize=12)
    plt.show()
