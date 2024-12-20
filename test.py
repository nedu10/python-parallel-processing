import os
# import sys

os.environ['PETSC_DIR'] = '/home/cifedior/Desktop/MASTERS/THESIS/CODE/petsc-3.14.1'
os.environ['PETSC_ARCH'] = 'arch-linux-c-opt'

from petsc4py import PETSc

A = PETSc.Mat().createAIJ(size=(n, n), csr=(row, col, data))
b = PETSc.Vec().createWithArray(F)
x = PETSc.Vec().createWithArray(np.zeros_like(F))

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.solve(b, x)
print(x.getArray())