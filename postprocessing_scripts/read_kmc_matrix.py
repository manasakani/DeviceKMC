import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, spmatrix
import matplotlib.pyplot as plt
import os

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def precondition_jacobi(
    matrices: list,
    rhss: list
) -> list:
    preconditioned_matrices = []
    preconditioned_rhs = []
    for matrix, rhs in zip(matrices, rhss):
        preconditioned_matrices.append(np.diag(1/np.diag(matrix)) @ matrix)
        preconditioned_rhs.append(np.diag(1/np.diag(matrix)) @ rhs)
    return preconditioned_matrices, preconditioned_rhs



def number_not_diagonal_dominant_rows(
    matrices: list,
    eps: float
):
    diagonal_dominant = []
    for matrix in matrices:
        diagonal_dominant.append(np.where(2*np.abs(np.diag(matrix)) - np.sum(
            np.abs(matrix), axis=1) <= -eps*np.abs(np.diag(matrix)) )[0].size)
    return diagonal_dominant

def assert_symmetric(
    matrices: list
):
    for matrix in matrices:
        assert np.allclose(matrix, matrix.T)

def cal_bandwidth_csr(A):
    bw = 0
    for i in range(A.shape[0]):
        for j in range(A.indptr[i], A.indptr[i+1]):
            bw_loc = np.abs(i-A.indices[j])
            if bw_loc > bw:
                bw = bw_loc
    return bw


def cal_mem(bw, n):
    memory = n+bw*(n-1 + n-bw)
    return memory


def cal_memT(bw, n):
    memory = n+bw*(2*n)
    return memory


def read_matrix(
    path: str,
    matrix_size: int,
    skiprows: int = 0
):
    data = np.loadtxt(path, skiprows=skiprows)

    data_vector = data[:, 1]
    row_indices = data[:, 0].astype(int) // matrix_size
    col_indices = data[:, 0].astype(int) % matrix_size

    matrix = coo_matrix((data_vector, (row_indices, col_indices)),
                        dtype=int, shape=(matrix_size, matrix_size))

    return matrix


def read_matrices_sparse(
    path: str,
    num_matrices: int
) -> list:
    sparse_matrices = []

    col_indices_path = path + "csrColIndices_step#"
    row_ptr_path = path + "csrRowPtr_step#"
    data_path = path + "csrValues_step#"
    for i in range(num_matrices):
        col_indices = np.loadtxt(col_indices_path + str(i) + ".txt")
        row_ptr = np.loadtxt(row_ptr_path + str(i) + ".txt")
        data = np.loadtxt(data_path + str(i) + ".txt")
        sparse_matrices.append(csr_matrix(
            (data, col_indices, row_ptr), dtype=int))
    return sparse_matrices


def read_matrices_dense(
    path: str,
    name: str,
    num_matrices: int,
    data_size: int
) -> list:
    matrices = []

    data_path = os.path.join(path, name)
    for i in range(num_matrices):
        data = np.loadtxt(data_path + str(i) + ".txt")

        assert data.size == data_size**2
        matrix = data.reshape((data_size, data_size))
        matrices.append(matrix)
    return matrices


def read_array(
    path: str,
    name: str,
    num_rhs: int,
    data_size: int
) -> list:
    data_path = path + name
    rhs = []
    for i in range(num_rhs):
        data = np.loadtxt(data_path + str(i) + ".txt")
        assert data.size == data_size
        rhs.append(data)
    return rhs


def plot_matrices(
    matrices: list,
    name: str,
    plotting_limit: float,
    path_images: str
):
    for i, matrix in enumerate(matrices):

        # if matrix is sparse
        if isinstance(matrix, spmatrix):
            matrix = matrix.toarray()

        if not is_invertible(matrix):
            print("Matrix " + str(i) + " is not invertible")

        matrix_abs = np.abs(matrix)
        matirx_norm = matrix_abs / np.max(matrix_abs)

        _, ax = plt.subplots()
        ax.matshow(np.log(plotting_limit+matirx_norm))
        plt.savefig(path_images + name + "_abs" + str(i) + ".png")
        plt.close()
        plt.spy(matirx_norm)
        plt.savefig(path_images + name + "_spy" + str(i) + ".png")


def write_matrices_csr(
    matrices: list,
    name: str,
    path: str
):
    for i, matrix in enumerate(matrices):
        matrix2 = csr_matrix(matrix)
        assert matrix2.has_sorted_indices
        with open(path + name + "_data" + str(i) + ".bin", "wb") as f:
            f.write(matrix2.data.reshape((1, -1)).tobytes())
        with open(path + name + "_indices" + str(i) + ".bin", "wb") as f:
            f.write(matrix2.indices.reshape((1, -1)).tobytes())
        with open(path + name + "_indptr" + str(i) + ".bin", "wb") as f:
            f.write(matrix2.indptr.reshape((1, -1)).tobytes())
        np.savetxt(path + name + "_data" + str(i) +
                   ".txt", matrix2.data.reshape((1, -1)))
        np.savetxt(path + name + "_indices" + str(i) + ".txt",
                   matrix2.indices.reshape((1, -1)), fmt="%i")
        np.savetxt(path + name + "_indptr" + str(i) + ".txt",
                   matrix2.indptr.reshape((1, -1)), fmt="%i")


def write_arrays(
    arrays: list,
    name: str,
    path: str
):
    for i, array in enumerate(arrays):
        np.savetxt(path + name + str(i) + ".txt", array.reshape((1, -1)))


def write_bin_arrays(
    arrays: list,
    name: str,
    path: str
):
    for i, array in enumerate(arrays):
        with open(path + name + str(i) + ".bin", "wb") as f:
            f.write(array.tobytes())
