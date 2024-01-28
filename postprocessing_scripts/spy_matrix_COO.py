import sys
import matplotlib.pyplot as plt
import numpy as np

def plot_coo_sparsity_pattern(row_indices, col_indices, filename):
    # Create a sparse matrix in COO format
    coo_matrix = np.zeros((max(row_indices) + 1, max(col_indices) + 1))
    for i, j in zip(row_indices, col_indices):
        coo_matrix[i, j] = 1  # Set the value to 1 for non-zero entries

    # Visualize the sparsity pattern
    plt.spy(coo_matrix, markersize=1)
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <coo_filename>")
        sys.exit(1)

    # Read COO format file
    filename_coo = sys.argv[1]
    data = np.loadtxt(filename_coo)

    # Split the data into row_indices, col_indices
    row_indices = data[:, 0].astype(int)
    col_indices = data[:, 1].astype(int)

    # Generate output JPG filename based on input filename
    output_filename = filename_coo.replace(".txt", "_sparsity_pattern.jpg")

    # Example usage
    plot_coo_sparsity_pattern(row_indices, col_indices, output_filename)
