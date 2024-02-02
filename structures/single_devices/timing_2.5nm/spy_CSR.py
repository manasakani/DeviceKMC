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

def print_nonzero_indices_and_values(matrix, label):
    indices = np.nonzero(matrix)
    for i, j in zip(indices[0], indices[1]):
        print(f"Non-zero element at position ({i}, {j}) in {label}: {matrix[i, j]}")

def visualize_and_save_sparsity_difference_log(csr_matrix_data_1, csr_matrix_data_2, folder_path, step_1, step_2):
    difference_matrix = csr_matrix_data_2 - csr_matrix_data_1

    # print("Non-zero indices and values in the difference matrix:")
    # print_nonzero_indices_and_values(difference_matrix, f'Difference (Step {step_2} - Step {step_1})')

    # Plot colors of the logarithm of the values
    plt.imshow(np.log1p(np.abs(difference_matrix.toarray())), cmap='viridis', interpolation='none', aspect='auto')

    plt.title(f'Logarithmic Sparsity Pattern Difference (Step {step_2} - Step {step_1})')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    filename = os.path.join(folder_path, f"log_sparsity_pattern_difference_step#{step_1}_to_step#{step_2}.jpg")
    plt.savefig(filename)
    plt.show()

def visualize_and_save_sparsity_difference(csr_matrix_data_1, csr_matrix_data_2, folder_path, step_1, step_2):
    difference_matrix = csr_matrix_data_2 - csr_matrix_data_1

    print("Non-zero indices and values in the difference matrix:")
    print_nonzero_indices_and_values(difference_matrix, f'Difference (Step {step_2} - Step {step_1})')

    plt.spy(difference_matrix, markersize=1)
    plt.title(f'Sparsity Pattern Difference (Step {step_2} - Step {step_1})')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    filename = os.path.join(folder_path, f"sparsity_pattern_difference_step#{step_1}_to_step#{step_2}.jpg")
    plt.savefig(filename)
    plt.show()

def are_matrices_equal(matrix1, matrix2, tolerance=1e-12):
    # Check if matrices have the same shape
    if matrix1.shape != matrix2.shape:
        return False

    # Get non-zero elements and their positions for both matrices
    nz_indices_1 = set(zip(matrix1.nonzero()[0], matrix1.nonzero()[1]))
    nz_indices_2 = set(zip(matrix2.nonzero()[0], matrix2.nonzero()[1]))

    # Check if non-zero elements match within the tolerance
    if not all(np.isclose(matrix1[i, j], matrix2[i, j], atol=tolerance) for i, j in nz_indices_1.intersection(nz_indices_2)):
        return False

    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_name>")
        sys.exit(1)

    folder_path = sys.argv[1]

    # Assuming steps 0 and 1 for illustration
    step_0 = 0
    step_1 = 1

    csr_matrix_data_0 = read_csr_matrix(folder_path, step_0)
    csr_matrix_data_1 = read_csr_matrix(folder_path, step_1)

    # Visualize and save sparsity pattern difference
    # visualize_and_save_sparsity_difference(csr_matrix_data_0, csr_matrix_data_1, folder_path, step_0, step_1)
    visualize_and_save_sparsity_difference_log(csr_matrix_data_0, csr_matrix_data_1, folder_path, step_0, step_1)

    # Check if matrices are equal
    if are_matrices_equal(csr_matrix_data_0, csr_matrix_data_1):
        print("Matrices are equal.")
    else:
        print("Matrices are not equal.")
        print("Indices and values where matrices differ:")
        indices_diff = np.nonzero(csr_matrix_data_0 != csr_matrix_data_1)
        for i, j in zip(indices_diff[0], indices_diff[1]):
            print(f"Position ({i}, {j}): Matrix 0 value = {csr_matrix_data_0[i, j]}, Matrix 1 value = {csr_matrix_data_1[i, j]}")

if __name__ == "__main__":
    main()
