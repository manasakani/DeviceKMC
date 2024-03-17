import sys
import os
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def read_csr_matrix(folder_path, step):
    filename_values = os.path.join(folder_path, f"csrValues_step#{step}.txt")
    filename_row = os.path.join(folder_path, f"csrRowPtr_step#{step}.txt")
    filename_col = os.path.join(folder_path, f"csrColIndices_step#{step}.txt")

    values = np.loadtxt(filename_values)
    row_ptr = np.loadtxt(filename_row, dtype=int)
    col_indices = np.loadtxt(filename_col, dtype=int)
    
    m = len(row_ptr) - 1
    nnz = len(values)
    csr_matrix_data = csr_matrix((values, col_indices, row_ptr), shape=(m, m))
    return csr_matrix_data

def visualize_and_save_sparsity(csr_matrix_data, folder_path, step):
    plt.spy(csr_matrix_data, markersize=1)
    plt.title('Sparsity Pattern')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    filename = os.path.join(folder_path, f"sparsity_pattern_step#{step}.jpg")
    plt.savefig(filename)
    plt.show()

def main():
    # if len(sys.argv) != 2:
    #     print("Usage: python script.py <folder_name>")
    #     sys.exit(1)

    folder_path = './'# sys.argv[1]
    step = 1

    # read the large sparse matrix
    csr_matrix_data = read_csr_matrix(folder_path, step)
    visualize_and_save_sparsity(csr_matrix_data, folder_path, step)
    vgkv

    # Convert the sparse matrix to a dense matrix
    dense_matrix = csr_matrix_data.toarray()

    # read the dense submatrix
    dense_submatrix_filename = os.path.join('./', 'T.txt')
    dense_submatrix = np.loadtxt(dense_submatrix_filename)

    print(np.shape(dense_submatrix))

    # read the insertion_indices
    insertion_indices_filename = os.path.join('./', 'insertion_indices.txt')
    insertion_indices = np.loadtxt(insertion_indices_filename, dtype=int)

    print(np.shape(insertion_indices))

    insertion_positions = insertion_indices +2

    # Add the dense submatrix to the corresponding positions in the dense matrix
    for i, row_index in enumerate(insertion_positions):
        for j, col_index in enumerate(insertion_positions):
            dense_matrix[row_index, col_index] += dense_submatrix[i, j]

    # Convert the dense matrix back to a sparse matrix
    updated_csr_matrix_data = csr_matrix(dense_matrix)

    diff_matrix = updated_csr_matrix_data - csr_matrix_data

    visualize_and_save_sparsity(updated_csr_matrix_data, folder_path, step)
    # visualize_and_save_sparsity(diff_matrix, folder_path, step)

if __name__ == "__main__":
    main()
