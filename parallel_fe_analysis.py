import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from sksparse.cholmod import cholesky
from scipy.io import savemat, loadmat
from multiprocessing import Pool
from timing import timing 

# @timing
def solve_single_force(args):
    """Solve FEA for a single column of the force matrix."""
    K, force_column, freedofs, positive_definite = args
    if positive_definite:
        spsolver = cholesky(K)
        return spsolver(force_column)
    else:
        return sp.linalg.spsolve(K, force_column)
    
@timing
def setup_analysis(iK, jK, sK, freedofs):
    """Setup FEA analysis."""
    # Construct the sparse stiffness matrix
    K = sp.csc_matrix((sK, (iK, jK)))
    K = K[np.ix_(freedofs, freedofs)]
    K = (K + K.T) / 2  # Ensure symmetry

    positive_definite = True
    try:
        spsolver = cholesky(K)  # Test if it is positive definite
    except Exception as e:
        print('Warning: Matrix is not positive definite:', e)
        positive_definite = False

    return K, positive_definite

def fe_analysis_parallel(U, iK, jK, sK, F, freedofs):
    """
    Perform finite element analysis in parallel.
    """
    K, positive_definite = setup_analysis(iK, jK, sK, freedofs)

    # Prepare arguments for parallel processing
    forces = F[freedofs, :]
    args = [(K, forces[:, i], freedofs, positive_definite) for i in range(F.shape[1])]

    # Solve in parallel
    with Pool() as pool:
        results = pool.map(solve_single_force, args)

    # Update U with results
    for i, result in enumerate(results):
        U[freedofs, i] = result

    return U

if __name__ == '__main__':
    def test0():
        from os.path import abspath, curdir, join
        mat_file = join(abspath(curdir), 'FEA_test1_py_copy.mat')
        mat_data = loadmat(mat_file)

        U_in = mat_data['U_in']
        iK = mat_data['iK']
        jK = mat_data['jK']
        sK = mat_data['sK']
        F = mat_data['F']
        freedofs = mat_data['freedofs']

        U_out = fe_analysis_parallel(U_in, iK[0], jK[0], sK[0], F, freedofs[0])

        print("U_out >> ", U_out)

        # savemat('FEA_test2_py.mat', {"U_out": U_out})

    test0()
