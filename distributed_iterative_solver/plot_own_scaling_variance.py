import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib  

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})

    number_of_measured_steps = 1
    base_path = "/usr/scratch/mont-fort17/almaeder/kmc_measurements/own_260/"
    libcomp_path = "/usr/scratch/mont-fort17/almaeder/kmc_measurements/own2/own_260/"
    images_path = "images/"

    reference_name = "solve"
    method_names_names = [
        "solve_allgatherv3",
        "solve_point_to_point3",
        "solve_custom_datatype2",
        "solve_gpu_packing3",
        "solve_ginkgo",
        "solve_petsc",
        "solve_hypre"
    ]
    matsize = 260

    sizes = [1, 2, 4, 8, 16, 32, 64]
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray"
    ]
    labels = [
        "Allgatherv",
        "Point to Point",
        "+ Custom Datatype",
        "+ GPU Packing",
        "Ginkgo",
        "PETSc",
        "HYPRE"
    ]
    paths = [
        base_path,
        base_path,
        base_path,
        base_path,
        base_path,
        base_path,
        base_path
    ]

    for step in range(1,number_of_measured_steps+1):
        reference_path = base_path + reference_name + "1" + "_" + str(step) + "_1_0.txt"
        reference_time = np.loadtxt(reference_path).flatten()

        median_reference_time1 = np.median(reference_time)

        reference_path = base_path + reference_name + "2" + "_" + str(step) + "_1_0.txt"
        reference_time = np.loadtxt(reference_path).flatten()

        median_reference_time2 = np.median(reference_time)
        reference_path = base_path + reference_name + "3" + "_" + str(step) + "_1_0.txt"
        reference_time = np.loadtxt(reference_path).flatten()

        median_reference_time3 = np.median(reference_time)
        reference_path = base_path + reference_name + "4" + "_" + str(step) + "_1_0.txt"
        reference_time = np.loadtxt(reference_path).flatten()

        median_reference_time4 = np.median(reference_time)

        median_reference_time = np.min([median_reference_time1, median_reference_time2, median_reference_time3, median_reference_time4])
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 9)
        for i, method_name in enumerate(method_names_names):

            times = [[] for j in range(len(sizes))]
            for j in range(len(sizes)):
                for k in range(sizes[j]):
                    times[j] += (np.loadtxt(paths[i] + method_name + "_" + str(step) + "_" + str(sizes[j]) + "_"+ str(k) +".txt").flatten()).tolist()
            
            if i == 4:
                print("d")
            times = [median_reference_time/np.array(times[j]) for j in range(len(sizes))]

        
            stds = []
            medians = []
            interval = []
            confidence = 0.95
            for j in range(len(sizes)):
                stds.append(np.std(times[j]))
                medians.append(np.median(times[j]))
                interval.append(st.t.interval(confidence=confidence, df=len(times[j])-1,
                        loc=np.median(times[j]),
                        scale=st.sem(times[j])))
                

            yerr_confidence = []
            for j in range(len(sizes)):
                yerr_confidence.append(np.copy(interval[j]))
                yerr_confidence[j][0] = -yerr_confidence[j][0] + medians[j]
                yerr_confidence[j][1] = yerr_confidence[j][1] - medians[j]

            eh_stds = []
            for j in range(len(sizes)):
                eh_stds.append(np.copy(interval[j]))
                eh_stds[j][0] = stds[j]
                eh_stds[j][1] = stds[j]


            yerr_confidence = np.array(yerr_confidence).T
            eh_stds = np.array(eh_stds).T
            x = np.array(sizes)
            ax.plot(x, medians, label=labels[i], color=colors[i], linestyle='dashed', linewidth=3)
            # ax.fill_between(x, inter_low, inter_high, alpha=0.2, color=colors[i])
            plt.errorbar(x, medians, yerr=np.squeeze(eh_stds), color=colors[i], capsize=10, barsabove=True, marker='x', linestyle='None', linewidth=3)

        #plt.plot(sizes, sizes, label="Linear scaling", color="black", linestyle='dashed', linewidth=1)
        plt.plot(sizes, np.ones((len(sizes))), color="black", linestyle='dashed', linewidth=0.5)

        # ax.set_yscale("log")
        # ax.set_title(
        #     "")
        ax.set_ylabel("Speedup")
        ax.set_xlabel("Nodes")
        ax.set_xticks(sizes)
        ax.set_xticklabels(sizes)
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, minor=False)
        ax.set_xticklabels(sizes, minor=False)
        plt.savefig(images_path + "own_scaling_variance.png", bbox_inches='tight', dpi=300)
