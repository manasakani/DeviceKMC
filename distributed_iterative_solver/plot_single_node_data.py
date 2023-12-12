import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})

    number_of_measured_steps = 1
    warmup = 10
    base_path = ""
    images_path = "images/"
    method_names_names = ["cg",
        "bicg",
        "gmres",
        "cg_jacobi",
        "bicg_jacobi",
        "gmres_jacobi",
        "cg_sor",
        "bicg_sor",
        "gmres_sor"]

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

    times = np.array(times)
    print(times.shape)
    # draw the box plot of the data
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    labels = method_names_names
    ax.boxplot(times.T, labels=labels, showfliers=False)
    ax.set_yscale("log")
    ax.set_title(
        "KMC over 500 Steps with PETSc")
    ax.set_ylabel("Time [s]")
    plt.savefig(images_path + "petsc_boxplot_CG_500.png", bbox_inches='tight', dpi=300)
