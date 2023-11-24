import numpy as np
from read_kmc_matrix import read_matrices_dense, plot_matrices
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # plot the condition numbers
    plt.rcParams.update({'font.size': 20})


    N = 9030
    N_left_tot = 1001
    N_interface = 7165
    N_right_tot = N - N_left_tot - N_interface
    Nsub = 5269  # second matrix
    assert N_right_tot == 864
    plotting_limit = 1e-10

    path_matrices = "/usr/scratch/attelas8/manasa/DeviceKMC/"
    num_matrices = 1
    
    path_images = "./"

    name_gpu = "testK_wDiag"
    matrices_gpu = read_matrices_dense(path_matrices, name_gpu, num_matrices, N)
    matrices_device_gpu = [matrix[N_left_tot:-N_right_tot,
                                  N_left_tot:-N_right_tot] for matrix in matrices_gpu]


    name_cpu = "testK_woDiag"
    matrices_cpu = read_matrices_dense(path_matrices, name_cpu, num_matrices, N)
    matrices_device_cpu = [matrix[N_left_tot:-N_right_tot,
                                    N_left_tot:-N_right_tot] for matrix in matrices_cpu]

    assum2 = np.zeros((N),dtype=float)
    for i in range(N):
        for j in range(N):
            assum2[i] += matrices_cpu[0][i,j]
    for i in range(N):
        assert matrices_cpu[0][i,i] == 0.0
	
    assumed_diagonal_values = np.sum(matrices_cpu[0],axis=1)
    assert np.allclose(assumed_diagonal_values, assum2)

    for i in range(N):
        print(np.abs(assumed_diagonal_values[i]+np.diag(matrices_gpu[0])[i]))
    for i in range(N):
        print(np.abs(assumed_diagonal_values[i]+np.diag(matrices_gpu[0])[i]))
        
    assert np.allclose(np.diag(matrices_gpu[0]), -assumed_diagonal_values)

    diag_gpu = np.copy(np.diag(matrices_gpu[0]))
    diag_device_gpu = np.copy(np.diag(matrices_device_gpu[0]))
    np.fill_diagonal(matrices_gpu[0], 0)
    np.fill_diagonal(matrices_device_gpu[0], 0)
   
    diag_cpu = np.copy(np.diag(matrices_cpu[0]))
    diag_device_cpu = np.copy(np.diag(matrices_device_cpu[0]))
    np.fill_diagonal(matrices_cpu[0], 0)
    np.fill_diagonal(matrices_device_cpu[0], 0)


    difference_dominance_gpu = []
    for matrix in matrices_device_gpu:
        difference_dominance_gpu.append(2*np.abs(np.diag(matrix)) - np.sum(np.abs(matrix), axis=1))
    
    difference_dominance_cpu = []
    for matrix in matrices_device_cpu:
        difference_dominance_cpu.append(2*np.abs(np.diag(matrix)) - np.sum(np.abs(matrix), axis=1))


    for i, data in enumerate(difference_dominance_gpu):
        # plot eigenvalues
        data[data>0] = 0.0
        fig, ax = plt.subplots()
        ax.plot(data)
        ax.set_xlabel("Row Index")
        ax.set_ylabel("Diagonally Dominance")
        # log y axis
        # ax.set_yscale('log')
        plt.savefig(path_images + "difference_dominance_gpu" + str(i) + ".png",
                    bbox_inches='tight', dpi=300)
        plt.close()

    for i, data in enumerate(difference_dominance_cpu):
        # plot eigenvalues
        data[data>0] = 0.0
        fig, ax = plt.subplots()
        ax.plot(data)
        ax.set_xlabel("Row Index")
        ax.set_ylabel("Diagonally Dominance")
        # log y axis
        # ax.set_yscale('log')
        plt.savefig(path_images + "difference_dominance_cpu" + str(i) + ".png",
                    bbox_inches='tight', dpi=300)
        plt.close()


    print(np.linalg.norm(matrices_cpu[0]- matrices_gpu[0])/
            np.linalg.norm(matrices_cpu[0]))
    print(np.linalg.norm(matrices_device_cpu[0]- matrices_device_gpu[0])/
            np.linalg.norm(matrices_device_cpu[0]))

    print(np.linalg.norm(diag_cpu - diag_gpu) / np.linalg.norm(diag_cpu))
    print(np.linalg.norm(diag_device_cpu - diag_device_gpu) / np.linalg.norm(diag_device_cpu))



    should_be_diag_gpu = -np.sum(matrices_gpu, axis=1)
    should_be_diag_cpu = -np.sum(matrices_cpu, axis=1)

    print(np.linalg.norm(should_be_diag_gpu - diag_gpu))
    print(np.linalg.norm(should_be_diag_cpu - diag_cpu))

