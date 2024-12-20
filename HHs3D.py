import numpy as np
from scipy.sparse import csr_matrix
import math

# Conversion Notes: The parameter 'nele' has been removed, after looking through app3.mlapp it
# it was found that nele = nelx * nely * nelz
def HHs3D(nelx: int, nely: int, nelz: int, rmin: float, ele: list[int]) -> tuple[csr_matrix, np.ndarray]:
    """
    Filter Preparation
 
    Parameters
    ----------
    nelx : int
        The number of x elements.
    nely : int
        The number of x elements.
    nelz : int
        The number of x elements.
    rmin : float
        The minimum of something.
    ele : list[int]
        Specifies a 2D slice of elements from the sparse matrix.

    Returns
    -------
    csr_matrix
        A sparse matrix
    np.ndarray
        A numpy matrix
    """
    nRmin = math.ceil(rmin)

    # Preallocate memory
    iH = np.ones((getMaxK(nelx, nely, nelz, nRmin)), dtype=int)
    jH = np.ones_like(iH)
    sH = np.zeros_like(iH, dtype=float)
    
    k = 0
    for k1 in range(nelz):
        for i1 in range(nelx):
            for j1 in range(nely):
                e1 = k1 * nelx * nely + i1 * nely + j1
                for k2 in range(max(k1 + 1 - nRmin, 0), min(k1 + nRmin, nelz)):
                    for i2 in range(max(i1 + 1 - nRmin, 0), min(i1 + nRmin, nelx)):
                        for j2 in range(max(j1 + 1 - nRmin, 0), min(j1 + nRmin, nely)):
                            e2 = k2 * nelx * nely + i2 * nely + j2
                            iH[k] = e1
                            jH[k] = e2
                            sH[k] = max(0, rmin - math.sqrt((i1 - i2)**2 + (j1 - j2)**2 + (k1 - k2)**2))
                            k += 1
    elements = np.ix_(ele, ele)

    H = csr_matrix((sH, (iH, jH)))
    H = H[elements]
    Hs = np.sum(H, axis=1)

    return H, Hs

def getMaxK(nelx: int, nely: int, nelz: int, rmin) -> int:
    k = 0
    for k1 in range(nelz):
        for i1 in range(nelx):
            for j1 in range(nely):
                for _ in range(max(k1 + 1 - rmin, 0), min(k1 + rmin, nelz)):
                    for _ in range(max(i1 + 1 - rmin, 0), min(i1 + rmin, nelx)):
                        for _ in range(max(j1 + 1 - rmin, 0), min(j1 + rmin, nely)):
                            k += 1
    return k    

if __name__ == '__main__':

    mat_data = loadmat('./matdata/HHs3Ddata.mat')

    nelx = mat_data['nelx']
    nely = mat_data['nely']
    nelz = mat_data['nelz']
    rmin = mat_data['rmin']
    ele = mat_data['ele']
    nele = mat_data['nele']

    print('nelx >> ', nelx[0, 0])
    print('nely >> ', nely[0, 0])
    print('nelz >> ', nelz[0, 0])
    print('rmin >> ', rmin[0, 0])
    print('ele >> ', ele.ravel())
    print('nele >> ', nele[0, 0])

    # Example call to the function
    ele = np.arange(1000)
    test1 = (10, 20, 10, 1.5, ele)    # works
    H, Hs = HHs3D(*test1)

    mdic = {"SAVEH": H, "SAVEHns": Hs}
    savemat("matlab_matrix.mat", mdic)

    print(Hs)
    print(Hs.shape)