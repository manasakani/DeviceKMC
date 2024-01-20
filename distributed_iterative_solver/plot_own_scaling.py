import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib  

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})

    number_of_measured_steps = 1
    warmup = 2
    base_path = "/usr/scratch/mont-fort17/almaeder/kmc_measurements/own_260/"
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
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
    ]
    labels = [
        "Allgatherv",
        "Nonblocking point-to-point",
        "+ fetch specific",
        "+ custom datatype",
        "+ GPU packing"
    ]


    for step in range(1,number_of_measured_steps+1):
        reference_path = base_path + reference_name + str(matsize) + "_" + str(step) + "_1_0_.txt"
        reference_time = np.loadtxt(reference_path)[warmup:]

        median_reference_time = np.median(reference_time)

        fig, ax = plt.subplots()
        fig.set_size_inches(16, 9)

        for i, method_name in enumerate(method_names_names):

            times = [[] for j in range(len(sizes))]
            for j in range(len(sizes)):
                for k in range(sizes[j]):
                    times[j] += (np.loadtxt(base_path + method_name + str(matsize) + "_" + str(step) + "_" + str(sizes[j]) + "_"+ str(k) +"_.txt")[warmup:]).tolist()
            
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
            ax.plot(x, medians, label=labels[i], color=colors[i], linestyle='dashed', linewidth=3)
            # ax.fill_between(x, inter_low, inter_high, alpha=0.2, color=colors[i])
            plt.errorbar(x, medians, yerr=np.squeeze(yer_fft), color=colors[i], capsize=10, barsabove=True, marker='x', linestyle='None', linewidth=3)

        plt.plot(sizes, sizes, label="Linear scaling", color="black", linestyle='dashed', linewidth=1)
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
        plt.savefig(images_path + "own_scaling.png", bbox_inches='tight', dpi=300)
