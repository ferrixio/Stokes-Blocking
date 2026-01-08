from StokesBuilder import *


def MatMult2Str(A:NDArray, B:NDArray):
    '''Questa funzione si occupa di calcolare manualmente A @ B e restituire una matrice
    di tipo stringa da stampare in .csv
    '''

    Asizes = A.shape
    Bsizes = B.shape
    ans = np.zeros((Asizes[0], Bsizes[1]), dtype=np.dtypes.StringDType)

    for i in range(Asizes[0]):
        for j in range(Bsizes[1]):
            row, col = A[i,:], B[:,j]

            ansStr = ''
            for k in range(len(row)):
                if row[k] not in (0,'') and col[k] not in (0,''):

                    if isinstance(row[k], float):
                        row[k] = f"{row[k]:.3f}"

                    if f"{row[k]}*{col[k]}"[0] == '-':
                        ansStr += f"{row[k]}*{col[k]}"
                    else:
                        ansStr += f"{row[k]}*{col[k]}+"

            ans[i,j] = f"({ansStr})"

    return ans



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

    # popolo la "matrice inversa" di stringhe
    # strInvA_xx = np.zeros(A_xx.shape, dtype=np.dtypes.StringDType)
    # for i in range(A_xx.shape[0]):
    #     for j in range(A_xx.shape[0]):
    #         strInvA_xx[i,j] = f'a({i},{j})'

    # t1 = MatMult2Str(Bx.T, strInvA_xx)
    # S = MatMult2Str(t1, Bx)

    S = np.hstack((A_xx, Bx))
    np.savetxt(f"Axx_Bx_{n}.csv", S, delimiter=';', fmt='%.3f')

    # BIG_Am1 = np.kron(np.eye(2), pinv(A_xx))

    # scrittura del complemento di Schur
    # S = BIG_B @ BIG_Am1 @ BIG_B.T
    # Sx = Bx.T @ pinv(A_xx) @ Bx
    # Sy = By.T @ pinv(A_xx) @ By
    # np.savetxt(f"BT_invA_B_x_n{n}.csv", Sx, delimiter=';', fmt='%16f')
    # np.savetxt(f"BT_invA_B_y_n{n}.csv", Sy, delimiter=';', fmt='%16f')