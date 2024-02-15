import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib  

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 36})
    matplotlib.rcParams['axes.linewidth'] =2
    matplotlib.rcParams['xtick.major.size'] = 10
    matplotlib.rcParams['xtick.major.width'] = 2
    matplotlib.rcParams['xtick.minor.size'] = 10
    matplotlib.rcParams['xtick.minor.width'] = 1
    matplotlib.rcParams['ytick.major.size'] = 10
    matplotlib.rcParams['ytick.major.width'] = 2
    matplotlib.rcParams['ytick.minor.size'] = 5
    matplotlib.rcParams['ytick.minor.width'] = 1

    linewidth = 3
    elinewidth = 4
    capsize = 10
    captick = 3

    number_of_measured_steps = 1
    base_path = "/usr/scratch/mont-fort17/almaeder/kmc_measurements/own_260/"
    libcomp_path = "/usr/scratch/mont-fort17/almaeder/kmc_measurements/own2/own_260/"
    images_path = "images/"
    sizes = [1, 2, 4, 8, 16, 32, 64]
    reference_name = "solve"
    matsize = 260


    method_names_names = [
        "solve_point_to_point3",
        "solve_custom_datatype2",
        "solve_gpu_packing3",
        "solve_petsc",
        "solve_hypre",
        "solve_ginkgo"
    ]
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:pink",
    ]
    labels = [
        "GPU Packing", 
        "Custom Datatype",               
        "Whole Piece",
        "PETSc",
        "HYPRE"
    ]

    method_names_names = [
        "solve_petsc",
        "solve_hypre",
        "solve_ginkgo"
    ]
    colors = [
        "tab:red",
        "tab:pink",
        "tab:purple",
    ]
    labels = [
        "PETSc",
        "HYPRE",
        "Ginkgo"
    ]

    # method_names_names = [
    #     "solve_point_to_point3",
    #     "solve_custom_datatype2",
    #     "solve_gpu_packing3",
    #     "solve_petsc",
    #     "solve_hypre",
    #     "solve_ginkgo",
    #     "solve_allgatherv3"
    # ]
    # colors = [
    #     "tab:orange",
    #     "tab:green",
    #     "tab:gray",
    #     "tab:red",
    #     "tab:pink",
    #     "tab:purple",
    #     "tab:blue"
    # ]
    # labels = [
    #     "Point to Point",
    #     "+ Custom Datatype",
    #     "+ GPU Packing",
    #     "PETSc",
    #     "HYPRE",
    #     "Ginkgo",
    #     "Allgatherv"
    # ]

    numerp = 1

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
        fig.set_size_inches(16, 12)
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
                

            yer_fft = []
            for j in range(len(sizes)):
                yer_fft.append(np.copy(interval[j]))
                yer_fft[j][0] = -yer_fft[j][0] + medians[j]
                yer_fft[j][1] = yer_fft[j][1] - medians[j]

            yer_fft = np.array(yer_fft).T
            x = np.array(sizes)
            ax.plot(x, medians, label=labels[i], color=colors[i], linestyle='dashed', linewidth=linewidth)
            # ax.fill_between(x, inter_low, inter_high, alpha=0.2, color=colors[i])
            plt.errorbar(x, medians, yerr=np.squeeze(yer_fft), color=colors[i], capsize=capsize, barsabove=True, marker='x', linestyle='None', linewidth=linewidth, elinewidth=elinewidth, capthick=captick)

        #plt.plot(sizes, sizes, label="Linear scaling", color="black", linestyle='dashed', linewidth=1)
        plt.plot(sizes, np.ones((len(sizes))), color="black", linestyle='dashed', linewidth=linewidth)

        # ax.set_yscale("log")
        # ax.set_title(
        #     "")
        ax.set_ylabel("Speedup")
        ax.set_xlabel("Nodes")
        ax.set_xticks(sizes)
        ax.set_xticklabels(sizes)
        ax.legend(fontsize="28", loc="upper left")
        ax.set_ylim(bottom=0,top=3.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, minor=False)
        ax.set_xticklabels(sizes, minor=False)
        plt.savefig(images_path + "own_scaling" +str(numerp)+".png", bbox_inches='tight', dpi=300)
