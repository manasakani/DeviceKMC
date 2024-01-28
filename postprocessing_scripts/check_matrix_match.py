import numpy as np
import sys

def check_matrix_match(matrix1, matrix2, tolerance=1e-10):
    mat1 = np.loadtxt(matrix1)
    mat2 = np.loadtxt(matrix2)

    # Check shape
    if mat1.shape != mat2.shape:
        print("Matrices have different shapes.")
        return False

    # Check element-wise match within tolerance
    mask = np.isclose(mat1, mat2, rtol=0, atol=tolerance)
    num_non_matching = np.sum(~mask)

    if num_non_matching > 0:
        print(f"Number of non-matching elements: {num_non_matching}")

        # Find and print the largest difference
        max_diff = np.max(np.abs(mat1 - mat2))
        print(f"Largest difference between matrices: {max_diff}")

        return False
    else:
        print("Matrices match within tolerance.")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 check_matrix_match.py matrix1.txt matrix2.txt")
        sys.exit(1)

    matrix1_file = sys.argv[1]
    matrix2_file = sys.argv[2]

    check_matrix_match(matrix1_file, matrix2_file)
