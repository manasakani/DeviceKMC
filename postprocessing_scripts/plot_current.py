import matplotlib.pyplot as plt
import sys

def main():
    if len(sys.argv) != 2:
        print("Missing output filename! Correct usage: $python3 plot_current.py output.log")
        sys.exit()

    outfile = sys.argv[1]

    print("reading from: " + outfile)
    
    # Initialize lists to store KMC time and current values
    kmc_time_list = [0.0]
    current_list = [0.0]

    reset_time = 0.0
    kmc_time = 0.0

    # Read the 'output.txt' file
    with open(outfile, "rt") as myfile:

        # Iterate through each line in the file
        for line in myfile:

            if "Applied Voltage =" in line:
                reset_time = kmc_time_list[-1]

            if "KMC time is:" in line:
                kmc_time = float(line.split(":")[1])
                kmc_time_list.append(kmc_time + reset_time)

            if "Current in uA:" in line:
                current = float(line.split()[3]) 
                current_list.append(current)

    # Plotting the data
    plt.plot(kmc_time_list, current_list, marker='o', markersize=5, linestyle='-')
    #plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('KMC Time')
    plt.ylabel('Current (uA)')
    plt.title('Current vs KMC Time')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('Current.jpg')

if __name__ == '__main__':
    main()
