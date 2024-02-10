import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

def plot_sparsity(matrix_filename):
    # Load the matrix from the text file
    matrix = np.loadtxt(matrix_filename)

    # Convert the matrix to a scipy sparse matrix
    sparse_matrix = coo_matrix(matrix)

    # Plot the sparsity pattern
    plt.spy(sparse_matrix, markersize=1)

    # Get the number of non-zero elements
    num_nonzero = sparse_matrix.getnnz()

    # Set the title with the number of non-zero elements
    plt.title(f'Sparsity Pattern (Nonzero: {num_nonzero})')

    # Save the figure with the input filename
    output_filename = matrix_filename.replace('.txt', '_sparsity_pattern.png')
    plt.savefig(output_filename)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 plot_sparsity_txt.py name-of-matrix-txt")
        sys.exit(1)

    matrix_filename = sys.argv[1]
    plot_sparsity(matrix_filename)
