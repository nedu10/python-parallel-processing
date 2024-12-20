import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from sksparse.cholmod import cholesky  # If sksparse is not available, this line will need adjustment

from scipy.io import savemat, loadmat
from timing import timing 

@timing
def fe_analysis(U, iK, jK, sK, F, freedofs):
    """
    Perform finite element analysis.
    
    Args:
        U (np.ndarray): Solution matrix to be updated.
        iK (np.ndarray): Row indices for the sparse matrix.
        jK (np.ndarray): Column indices for the sparse matrix.
        sK (np.ndarray): Data values for the sparse matrix.
        F (np.ndarray): Force matrix.
        freedofs (np.ndarray): Indices of free degrees of freedom.
    """
    # Construct the sparse stiffness matrix
    K = sp.csc_matrix((sK, (iK, jK)))
    K = K[np.ix_(freedofs, freedofs)]
    K = (K + K.T) / 2  # Ensure symmetry

    print("here >> ", K.shape)

    positive_definite = True
    try:
        spsolver = cholesky(K)
    except Exception as e:
        print('Warning: Matrix is not positive definite:', e)
        positive_definite = False 

    print("positive_definite >> ", positive_definite)

    # Extract forces for the free degrees of freedom
    Forces = F[freedofs, :]

    for i in range(F.shape[1]):
        if positive_definite:
            U[freedofs, i] = spsolver(Forces[:, i])
        else:
            U[freedofs, i] = sp.linalg.spsolve(K, Forces[:, i])

    return U



if __name__ == '__main__':

    def test0():
        from os.path import abspath, curdir, join
        mat_file = join(abspath(curdir), 'FEA_test1_py_copy.mat')
        mat_data = loadmat(mat_file)

        print("mat_data >> ", mat_data)

        U_in = mat_data['U_in']
        U = mat_data['U']
        iK = mat_data['iK']
        jK = mat_data['jK']
        sK = mat_data['sK']
        F = mat_data['F']
        freedofs = mat_data['freedofs']
        Ee = mat_data['Ee']

        print("iK >> ", iK[0].shape)
        print("jK >> ", jK[0].shape)
        print("sK >> ", sK[0].shape)
        print("F >> ", F.shape)
        print("freedofs >> ", freedofs[0].shape)

        U_out = fe_analysis(U_in, iK[0], jK[0], sK[0], F, freedofs[0])

        print("U_out >> ", U_out)

        mdic = {
            "U_out": U_out
        }
        savemat('FEA_test2_py.mat', mdic)


    test0()
