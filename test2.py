import cupy as cp

try:
    cp.linalg.cholesky(cp.random.rand(10, 10))
    print("CuPy is properly using the GPU.")
except Exception as e:
    print("Error:", e)
