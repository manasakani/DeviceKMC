import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})

    number_of_measured_steps = 1
    warmup = 2
    base_path = ""
    images_path = "images/"

    reference_name = "solve_cg"
    method_names_names = [
        "solve_cg_allgatherv_mpi",
        "solve_cg_nonblocking_point_to_point",
        "solve_cg_nonblocking_point_to_point_fetch_specific",
        "solve_cg_nonblocking_point_to_point_fetch_specific_custom_datatype",
        "solve_cg_nonblocking_point_to_point_fetch_specific_gpu_packing"]
    matsize = 260

    sizes = [1, 2, 4, 8]

    for step in range(number_of_measured_steps):
        reference_path = base_path + reference_name + str(matsize) + " " + str(step) + "_1_0.txt"
        reference_time = np.loadtxt(reference_path)[warmup:]

        fig, ax = plt.subplots()
        fig.set_size_inches(16, 9)

        for i, method_name in enumerate(method_names_names):

            times = [[] for j in range(len(sizes))]

            medians = []



            for method_name in method_names_names:
                ksp_iterations = np.loadtxt(base_path + method_name + "_iterations.txt")

                fig, ax = plt.subplots()
                ax.scatter(np.arange(ksp_iterations.size), ksp_iterations)
                ax.set_xlabel("KMC Step")
                ax.set_ylabel("CG Steps")
                ax.set_title(
                    method_name + "\n" +
                    "Mean: " + str(int(np.mean(ksp_iterations))) + "\n" +
                    "Median: " + str(int(np.median(ksp_iterations))))
                plt.savefig(images_path + "petsc_" + method_name + "_iterations.png",
                bbox_inches='tight', dpi=300)
                plt.close()

            times = [[] for i in range(len(method_names_names))]
            for i, method_name in enumerate(method_names_names):
                for measurement in range(number_of_measured_steps):
                    times[i] += (np.loadtxt(base_path + method_name + "_times" +str(measurement) +"_1_0.txt")[warmup:]).tolist()

        # ax.set_yscale("log")
        # ax.set_title(
        #     "")
        ax.set_ylabel("Time [s]")
        ax.set_xlabel("Nodes")
        plt.savefig(images_path + "own_scaling.png", bbox_inches='tight', dpi=300)
