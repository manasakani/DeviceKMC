import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, spmatrix
import matplotlib.pyplot as plt
import os

col_indices_path = "csrColIndices_step#1.txt"
row_ptr_path = "csrRowPtr_step#1.txt"
data_path = "csrValues_step#1.txt"

col_indices = np.loadtxt(col_indices_path)
row_ptr = np.loadtxt(row_ptr_path)
data = np.loadtxt(data_path)

sparse_matrix = csr_matrix((data, col_indices, row_ptr), dtype=float)
plt.spy(sparse_matrix, markersize=0.5)
plt.title('Sparsity Pattern')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.savefig('sparsity_pattern.png')
