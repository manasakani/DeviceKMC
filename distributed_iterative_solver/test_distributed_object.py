import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":

    matrix_size = 2*3651
    size = 5
    rows_per_rank = [1461, 1461, 1460, 1460, 1460]
    nnz_per_rank = [[34653, 1670], [33345, 1701, 1670], [33430, 1572, 1701], [34832, 2765, 1572], [35008, 2765]]
    data_path = "/usr/scratch/mont-fort17/almaeder/kmc_"+str(7)+"k/split_matrix"
    image_path = "/home/sem23f28/Documents/DeviceKMC/distributed_iterative_solver/images"

    for rank in range(size):
        for i, nnz in enumerate(nnz_per_rank[rank]):
            row_ptr_path = os.path.join(data_path, "A_row_ptr" + str(size) + "_"+ str(rank) + "_"+ str(i) +".bin")
            col_indices_path = os.path.join(data_path, "A_col_indices" + str(size) + "_"+ str(rank) + "_"+ str(i) +".bin")
            row_ptr = np.fromfile(row_ptr_path, dtype=np.int32)
            col_indices = np.fromfile(col_indices_path, dtype=np.int32)
            data = np.ones(nnz)
            matrix = csr_matrix(
                (data, col_indices, row_ptr), dtype=float, shape=(rows_per_rank[rank],rows_per_rank[(rank+i)%size])).toarray()
            plt.spy(matrix)
            plt.savefig(os.path.join(image_path, "A" + str(size) + "_"+ str(rank) + "_"+ str(i) + ".png"))