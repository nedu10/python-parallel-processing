

import numpy as np
import math

# Conversion Notes: The parameter 'nele' has been removed, after looking through app3.mlapp it
# it was found that nele = nelx * nely * nelz
def smooth3D_convert_to_elements(nelx: int, nely: int, nelz: int, ngrid: int, vxPhys: np.ndarray, xgnew: np.ndarray):

    Terr = 0

    Tm = []

    for nk in range(nelz):
        for ni in range(nelx):
            for nj in range(nely):
                ne = (nk * nelx * nely) + (ni * nely) + nj
                for nk1 in range(ngrid * nk, ngrid * (nk + 1) + 1):
                    for ni1 in range(ngrid * ni, ngrid * (ni + 1) + 1):
                        for nj1 in range(ngrid * nj, ngrid * (nj + 1) + 1):
                            Tm.append(xgnew[nj1, ni1, nk1])
                            vxPhys[ne] += xgnew[nj1, ni1, nk1]
                if np.min(Tm) > 0.001 and np.max(Tm) < 1:
                    Terr += 1
                Tm.clear()
