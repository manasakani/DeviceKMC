import matplotlib.pyplot as plt
import sys

def main():
    if len(sys.argv) != 2:
        print("Missing output filename! Correct usage: $python3 plot_current.py output.log")
        sys.exit()

    outfile = sys.argv[1]

    print("reading from: " + outfile)
    
    # Initialize lists to store KMC time and current values
    kmc_voltage_list = []
    current_list = []
    current_track = []
    counter = 0

    # Read the 'output.txt' file
    with open(outfile, "rt") as myfile:

        # Iterate through each line in the file
        for line in myfile:

            if "Applied Voltage =" in line:
                kmc_voltage = float(line.split()[3])
                kmc_voltage_list.append(kmc_voltage)

                if counter > 0:
                    current_list.append(current_track[-1])

            if "Current in uA:" in line:
                current = float(line.split()[3]) 
                current_track.append(current)
                counter += 1

    current_list.append(current_track[-1])

    # Plotting the data
    plt.plot(kmc_voltage_list, current_list, marker='o', markersize=5, linestyle='-')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('Current (uA)')
    plt.title('Current vs applied voltage')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('Current.jpg')

if __name__ == '__main__':
    main()
