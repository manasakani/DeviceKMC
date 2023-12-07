import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
import os

def write_matrices_csr(
    matrix: csr_matrix,
    path: str,
    name: str,
):
    assert matrix.has_sorted_indices
    with open(path + name + "_data.bin", "wb") as f:
        f.write(matrix.data.tobytes())
    with open(path + name + "_indices.bin", "wb") as f:
        f.write(matrix.indices.tobytes())
    with open(path + name + "_indptr.bin", "wb") as f:
        f.write(matrix.indptr.tobytes())



def write_array(
    array: np.ndarray,
    name: str,
    path: str
):
    with open(path + name + ".bin", "wb") as f:
        f.write(array.tobytes())


if __name__ == "__main__":
    save_path = "../test_data/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    matrix_size = 10
    test_matrix = sparse.random(matrix_size, matrix_size, density=1)
    test_matrix = test_matrix.tocsr()
    test_matrix = test_matrix + test_matrix.T

    vector_x = np.random.rand(matrix_size)
    vector_y = test_matrix @ vector_x

    vector_rhs = np.random.rand(matrix_size)
    vector_sol = sparse.linalg.spsolve(test_matrix, vector_rhs)

    write_matrices_csr(test_matrix, save_path, "matrix")
    write_array(vector_x, "x", save_path)
    write_array(vector_y, "y", save_path)

    path_matrix_parameters = save_path + "matrix_parameters.txt"
    with open(path_matrix_parameters, "w") as f:
        f.write(str(matrix_size) + "\n")
