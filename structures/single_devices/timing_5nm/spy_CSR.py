import sys
import os
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def read_csr_matrix(folder_path, step):
    print("here1")

    filename_values = os.path.join(folder_path, f"csrValues_step#{step}.txt")
    filename_row = os.path.join(folder_path, f"csrRowPtr_step#{step}.txt")
    filename_col = os.path.join(folder_path, f"csrColIndices_step#{step}.txt")
    print("here2")


    values = np.loadtxt(filename_values) +1 
    row_ptr = np.loadtxt(filename_row, dtype=int)
    col_indices = np.loadtxt(filename_col, dtype=int)
    print("here3")
    
    m = len(row_ptr) - 1
    nnz = len(values)
    csr_matrix_data = csr_matrix((values, col_indices, row_ptr), shape=(m, m))
    print("here4")
    return csr_matrix_data

def visualize_and_save_sparsity(csr_matrix_data, folder_path, step):
    print("here5")
    plt.spy(csr_matrix_data.toarray(), markersize=1)
    print("here6")
    plt.title('Sparsity Pattern')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    print("here7")

    plt.savefig("temp.jpg")
    sdfg

    filename = os.path.join(folder_path, f"sparsity_pattern_step#{step}.jpg")
    plt.savefig(filename)
    # plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_name>")
        sys.exit(1)

    folder_path = sys.argv[1]

    # Assuming step 0 for illustration
    step = 0

    csr_matrix_data = read_csr_matrix(folder_path, step)
    visualize_and_save_sparsity(csr_matrix_data, folder_path, step)

if __name__ == "__main__":
    main()
