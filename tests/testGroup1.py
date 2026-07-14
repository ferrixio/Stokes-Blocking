'''
Group 1: Krylov tests

mu(x,y) = 1 constant

We are testing:
    1. Clustering of the preconditioners of A_n and B_n
    2. GMRES convergence
    3. PGMRES convergence

(P)GMRES is tested with the following right-hand sides:
    a. rhs = 1
    b. rhs = g(t1,t2)*h^2   where t1,t2 are samplings of [-pi, pi]^2 and h = dx
    c. rhs = rng(t1,t2)     where rng is a random sampling of [-pi, pi]^2
'''

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.StokesBuilder import *

if __name__ == "__main__":

    n = 24                       # order of division
    coeffMu = CreateMu(n, 1, 0)  # viscosity coefficients


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


    # Preconditioner construction
    PRECOND_Axx = Create_Axx_Precond(A_xx, n)
    PRECOND_A_FULL = np.kron(np.eye(2), PRECOND_Axx)
    invPA = np.kron(np.eye(2), inv(PRECOND_Axx))

    S = BIG_B @ invPA @ BIG_B.T
    PP = np.block([
        [PRECOND_A_FULL, np.zeros((BIG_A.shape[0], S.shape[1]))],
        [np.zeros((S.shape[0], BIG_A.shape[1])), S]
    ])
    SCHUR_P = pinv(PP)

    # save("precond_test1_n32.npy", SCHUR_P)
    # SCHUR_P = load('precond_test1_n32.npy')

    # inv(P)*M clustering
    sing = np.linalg.svd(np.matmul(SCHUR_P, SUPER_A), compute_uv=False)
    xx = np.linspace(0,1,len(sing))

    plt.figure(1)
    plt.semilogy(xx[:-1], sing[:-1], 'x', color='red', label=r'$\sigma_j(S_n^{-1}M_n)$')
    plt.axhline(y=1, color='black', linestyle='-')
    plt.legend(fontsize=14)
    plt.show()

    # GMRES test
    rhs = np.ones((SUPER_A.shape[0], 1))
    # rhs = CreateUniformSampling(SUPER_A.shape[0], n)
    # rhs = CreateRandomSampling(SUPER_A.shape[0], n)
    # rhs = loadtxt(f'test_n{n}.txt')

    # PGMRES test
    counter11 = krylov_counter()
    x, err11 = gmres(SUPER_A, rhs, M=SCHUR_P, callback=counter11)