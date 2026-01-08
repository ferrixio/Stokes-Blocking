'''
COMMENT: Primo gruppo di test per stokes.

mu = 1 ovunque

Vengono testate le seguenti cose:
    1. Clustering a 1 del precondizionatore di A e di B
    2. GMRES
    3. PGMRES

(P)GMRES viene testato su tre sistemi lineari diversi:
    a. rhs = 1
    b. rhs = g(t1,t2)*h^2,  dove t1,t2 sono un sampling di [-pi, pi]^2 e h = dx
    c. rhs = rng(t1,t2),    dove rng è un sampling randomico di [-pi, pi]^2
'''

from src.StokesBuilder import *

if __name__ == "__main__":

    n = 8                       # ordine della divisione
    coeffMu = CreateMu(n, 1, 0) # ottenimento dei coefficienti di mu


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


    # STEP: costruzione precondizionatore
    PRECOND_Axx = Create_Axx_Precond(A_xx, n)
    PRECOND_A_FULL = np.kron(np.eye(2), PRECOND_Axx)
    invPA = np.kron(np.eye(2), inv(PRECOND_Axx))

    # PRECOND_B_FULL = Create_B_Precond(Bx, By, n)
    # invB = pinv(BIG_B)
    # invBT = pinv(BIG_B.T)
    # invS = invBT @ PRECOND_A_FULL @ invB
    # SCHUR_P = np.block([
    #     [invPA, np.zeros((BIG_A.shape[0], invS.shape[1]))],
    #     [np.zeros((invS.shape[0], BIG_A.shape[1])), invS]
    # ])

    S = BIG_B @ invPA @ BIG_B.T
    PP = np.block([
        [PRECOND_A_FULL, np.zeros((BIG_A.shape[0], S.shape[1]))],
        [np.zeros((S.shape[0], BIG_A.shape[1])), S]
    ])
    SCHUR_P = pinv(PP)

    # save("precond_test1_n32.npy", SCHUR_P)
    # SCHUR_P = load('precond_test1_n32.npy')


    # STEP: studio del clustering
    # sing = np.linalg.svd(np.matmul(SCHUR_P, SUPER_A), compute_uv=False)
    # xx = np.linspace(0,1,len(sing))

    # plt.figure(1)
    # plt.semilogy(xx, sing, 'x', color='red', markersize=3, label=r'$S_n^{-1}*A_n$')
    # plt.axhline(y=1, color='black', linestyle='-', label='y=1')
    # plt.legend(fontsize=12)
    # plt.show()


    # STEP: GMRES in this economy?
    rhs = np.ones((SUPER_A.shape[0], 1))
    # rhs = CreateUniformSampling(SUPER_A.shape[0], n)
    # rhs = CreateRandomSampling(SUPER_A.shape[0], n)
    # rhs = loadtxt(f'test_n{n}.txt')

    #sistemi precondizionati
    counter11 = krylov_counter()
    x, err11 = gmres(SUPER_A, rhs, M=SCHUR_P, callback=counter11)