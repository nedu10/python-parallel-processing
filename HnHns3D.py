import numpy as np
from scipy.sparse import csr_matrix
from math import ceil, sqrt

# Conversion Notes: Changed indexing to start at zero in for loops.
def HnHns3D(nelx: int, nely: int, nelz: int, rnmin: float) -> tuple[csr_matrix, np.ndarray]:
        """
        Compute the Hn and Hns matrices for 3D topology optimization.

        Parameters:
        nelx (int): Number of elements in the x-direction.
        nely (int): Number of elements in the y-direction.
        nelz (int): Number of elements in the z-direction.
        rnmin (float): Minimum radius.

        Returns:
        Tuple[csr_matrix, np.ndarray]: Filter matrix
        """
        rmin = ceil(rnmin)

        iH = np.ones((getMaxK(nelx, nely,nelz, rmin)), dtype=int)
        jH = np.ones_like(iH)
        sH = np.zeros_like(iH, dtype=float)

        k = 0
        elex, eley, elez = np.meshgrid(np.arange(1.5, nelx + 1.5), np.arange(1.5, nely + 1.5), np.arange(1.5, nelz + 1.5))
        for k1 in range(nelz + 1):
            for i1 in range(nelx + 1):
                for j1 in range(nely + 1):
                    e1 = k1 * (nelx + 1) * (nely + 1) + i1 * (nely + 1) + j1
                    for k2 in range(max(k1 - rmin, 0), min(k1 + rmin, nelz)):
                        for i2 in range(max(i1 - rmin, 0), min(i1 + rmin, nelx)):
                            for j2 in range(max(j1 - rmin, 0), min(j1 + rmin, nely)):
                                e2 = k2 * nelx * nely + i2 * nely + j2
                                iH[k] = e1
                                jH[k] = e2
                                sH[k] = max(0, rnmin - sqrt((i1 + 1 - elex[j2,i2,k2])**2+(j1 + 1 - eley[j2,i2,k2])**2+(k1 + 1 - elez[j2,i2,k2])**2))
                                k += 1
        Hn = csr_matrix((sH, (iH, jH)))
        Hns = np.sum(Hn, axis=1)

        return Hn, Hns

def getMaxK(nelx, nely, nelz, rmin) -> int:
    k = 0
    for k1 in range(nelz + 1):
        for i1 in range(nelx + 1):
            for j1 in range(nely + 1):
                for _ in range(max(k1 - rmin, 0), min(k1 + rmin, nelz)):
                    for _ in range(max(i1 - rmin, 0), min(i1 + rmin, nelx)):
                        for _ in range(max(j1 - rmin, 0), min(j1 + rmin, nely)):
                             k += 1
    return k

if __name__ == '__main__':
     H, Hns = HnHns3D(10, 20, 10, 1.5)
    #  print(H)
    #  print(H.shape)
     print(Hns)
     print(Hns.shape)