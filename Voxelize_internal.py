import numpy as np
from typing import List, Tuple

def VOXELISEinternal(testx: np.ndarray , testy: np.ndarray, testz: np.ndarray, meshXYZ: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Internal function to compute voxelization along a specific axis.
    """
    np.seterr(divide='ignore')
    OUTPUT = np.zeros(testx.shape[0], dtype=np.bool_)
    correctionLIST = []

    meshZmin = np.min(meshXYZ[:, 2, :])
    meshZmax = np.max(meshXYZ[:, 2, :])
    meshXYZmin = np.min(meshXYZ, axis=2)
    meshXYZmax = np.max(meshXYZ, axis=2)

    for loop in range(len(OUTPUT)):
        possibleCROSSLISTy = np.where((testy[loop] - meshXYZmin[:, 1]) * (meshXYZmax[:, 1] - testy[loop]) > 0)[0]
        possibleCROSSLISTx = np.where((testx[loop] - meshXYZmin[possibleCROSSLISTy, 0]) * (meshXYZmax[possibleCROSSLISTy, 0] - testx[loop]) > 0)[0]
        possibleCROSSLIST = possibleCROSSLISTy[possibleCROSSLISTx]

        if len(possibleCROSSLIST) > 0:
            facetCROSSLIST = []
            for loopCHECKFACET in possibleCROSSLIST:
                Y1predicted = meshXYZ[loopCHECKFACET, 1, 1] - ((meshXYZ[loopCHECKFACET, 1, 1] - meshXYZ[loopCHECKFACET, 1, 2]) * (meshXYZ[loopCHECKFACET, 0, 1] - meshXYZ[loopCHECKFACET, 0, 0]) / (meshXYZ[loopCHECKFACET, 0, 1] - meshXYZ[loopCHECKFACET, 0, 2]))
                YRpredicted = meshXYZ[loopCHECKFACET, 1, 1] - ((meshXYZ[loopCHECKFACET, 1, 1] - meshXYZ[loopCHECKFACET, 1, 2]) * (meshXYZ[loopCHECKFACET, 0, 1] - testx[loop]) / (meshXYZ[loopCHECKFACET, 0, 1] - meshXYZ[loopCHECKFACET, 0, 2]))
                if not ((Y1predicted > meshXYZ[loopCHECKFACET, 1, 0] and YRpredicted > testy[loop]) or (Y1predicted < meshXYZ[loopCHECKFACET, 1, 0] and YRpredicted < testy[loop]) or (meshXYZ[loopCHECKFACET, 1, 1] - meshXYZ[loopCHECKFACET, 1, 2]) * (meshXYZ[loopCHECKFACET, 0, 1] - testx[loop]) == 0):
                    continue

                Y2predicted = meshXYZ[loopCHECKFACET, 1, 2] - ((meshXYZ[loopCHECKFACET, 1, 2] - meshXYZ[loopCHECKFACET, 1, 0]) * (meshXYZ[loopCHECKFACET, 0, 2] - meshXYZ[loopCHECKFACET, 0, 1]) / (meshXYZ[loopCHECKFACET, 0, 2] - meshXYZ[loopCHECKFACET, 0, 0]))
                YRpredicted = meshXYZ[loopCHECKFACET, 1, 2] - ((meshXYZ[loopCHECKFACET, 1, 2] - meshXYZ[loopCHECKFACET, 1, 0]) * (meshXYZ[loopCHECKFACET, 0, 2] - testx[loop]) / (meshXYZ[loopCHECKFACET, 0, 2] - meshXYZ[loopCHECKFACET, 0, 0]))
                if not ((Y2predicted > meshXYZ[loopCHECKFACET, 1, 1] and YRpredicted > testy[loop]) or (Y2predicted < meshXYZ[loopCHECKFACET, 1, 1] and YRpredicted < testy[loop]) or (meshXYZ[loopCHECKFACET, 1, 2] - meshXYZ[loopCHECKFACET, 1, 0]) * (meshXYZ[loopCHECKFACET, 0, 2] - testx[loop]) == 0):
                    continue

                Y3predicted = meshXYZ[loopCHECKFACET, 1, 0] - ((meshXYZ[loopCHECKFACET, 1, 0] - meshXYZ[loopCHECKFACET, 1, 1]) * (meshXYZ[loopCHECKFACET, 0, 0] - meshXYZ[loopCHECKFACET, 0, 2]) / (meshXYZ[loopCHECKFACET, 0, 0] - meshXYZ[loopCHECKFACET, 0, 1]))
                YRpredicted = meshXYZ[loopCHECKFACET, 1, 0] - ((meshXYZ[loopCHECKFACET, 1, 0] - meshXYZ[loopCHECKFACET, 1, 1]) * (meshXYZ[loopCHECKFACET, 0, 0] - testx[loop]) / (meshXYZ[loopCHECKFACET, 0, 0] - meshXYZ[loopCHECKFACET, 0, 1]))
                if not ((Y3predicted > meshXYZ[loopCHECKFACET, 1, 2] and YRpredicted > testy[loop]) or (Y3predicted < meshXYZ[loopCHECKFACET, 1, 2] and YRpredicted < testy[loop]) or (meshXYZ[loopCHECKFACET, 1, 0] - meshXYZ[loopCHECKFACET, 1, 1]) * (meshXYZ[loopCHECKFACET, 0, 0] - testx[loop]) == 0):
                    continue

                facetCROSSLIST.append(loopCHECKFACET)

            gridCOzCROSS = np.zeros(len(facetCROSSLIST))
            for loopFINDZ in facetCROSSLIST:
                planecoA = meshXYZ[loopFINDZ, 1, 0] * (meshXYZ[loopFINDZ, 2, 1] - meshXYZ[loopFINDZ, 2, 2]) + meshXYZ[loopFINDZ, 1, 1] * (meshXYZ[loopFINDZ, 2, 2] - meshXYZ[loopFINDZ, 2, 0]) + meshXYZ[loopFINDZ, 1, 2] * (meshXYZ[loopFINDZ, 2, 0] - meshXYZ[loopFINDZ, 2, 1])
                planecoB = meshXYZ[loopFINDZ, 2, 0] * (meshXYZ[loopFINDZ, 0, 1] - meshXYZ[loopFINDZ, 0, 2]) + meshXYZ[loopFINDZ, 2, 1] * (meshXYZ[loopFINDZ, 0, 2] - meshXYZ[loopFINDZ, 0, 0]) + meshXYZ[loopFINDZ, 2, 2] * (meshXYZ[loopFINDZ, 0, 0] - meshXYZ[loopFINDZ, 0, 1])
                planecoC = meshXYZ[loopFINDZ, 0, 0] * (meshXYZ[loopFINDZ, 1, 1] - meshXYZ[loopFINDZ, 1, 2]) + meshXYZ[loopFINDZ, 0, 1] * (meshXYZ[loopFINDZ, 1, 2] - meshXYZ[loopFINDZ, 1, 0]) + meshXYZ[loopFINDZ, 0, 2] * (meshXYZ[loopFINDZ, 1, 0] - meshXYZ[loopFINDZ, 1, 1])
                planecoD = - meshXYZ[loopFINDZ, 0, 0] * (meshXYZ[loopFINDZ, 1, 1] * meshXYZ[loopFINDZ, 2, 2] - meshXYZ[loopFINDZ, 1, 2] * meshXYZ[loopFINDZ, 2, 1]) - meshXYZ[loopFINDZ, 0, 1] * (meshXYZ[loopFINDZ, 1, 2] * meshXYZ[loopFINDZ, 2, 0] - meshXYZ[loopFINDZ, 1, 0] * meshXYZ[loopFINDZ, 2, 2]) - meshXYZ[loopFINDZ, 0, 2] * (meshXYZ[loopFINDZ, 1, 0] * meshXYZ[loopFINDZ, 2, 1] - meshXYZ[loopFINDZ, 1, 1] * meshXYZ[loopFINDZ, 2, 0])
                
                if abs(planecoC) < 1e-14:
                    planecoC = 0

                gridCOzCROSS[facetCROSSLIST.index(loopFINDZ)] = (-planecoD - planecoA * testx[loop] - planecoB * testy[loop]) / planecoC

            if len(gridCOzCROSS) == 0:
                continue

            gridCOzCROSS = gridCOzCROSS[(gridCOzCROSS >= meshZmin - 1e-12) & (gridCOzCROSS <= meshZmax + 1e-12)]
            gridCOzCROSS = np.round(gridCOzCROSS * 1e10) / 1e10
            gridCOzCROSS = np.unique(gridCOzCROSS)

            if len(gridCOzCROSS) % 2 == 0:
                for loopASSIGN in range(len(gridCOzCROSS) // 2):
                    voxelsINSIDE = (testz[loop] > gridCOzCROSS[2 * loopASSIGN] and testz[loop] < gridCOzCROSS[2 * loopASSIGN + 1])
                    OUTPUT[loop] = voxelsINSIDE
                    if voxelsINSIDE:
                        break
            elif len(gridCOzCROSS) != 0:
                correctionLIST.append(loop)
    np.seterr(divide='warn')

    return OUTPUT, np.array(correctionLIST, dtype=int)