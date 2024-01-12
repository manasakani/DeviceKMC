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

            if "Current [uA]:" in line:
                current = float(line.split()[-1]) 
                current_track.append(current)
                counter += 1

    current_list.append(current_track[-1])

    plt.figure(figsize=(5, 4))

    # # Plotting the data
    # plt.plot(kmc_voltage_list, current_list, marker='o', markersize=5, linestyle='-')
    # #plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Applied Voltage (V)')
    # plt.ylabel('Current (uA)')
    # plt.title('Current vs applied voltage')

    print(kmc_voltage_list)
    print(current_list)

    # Plotting the data on the left y-axis (linear scale)
    plt.plot(kmc_voltage_list, current_list, marker='o', markersize=5, linestyle='-', label='Linear Scale')
    plt.yscale('linear')  # Set the left y-axis to linear scale
    plt.ylim(-1, 10)
    plt.xlim(0, 9)

    # Creating a secondary y-axis with log scale
    ax2 = plt.gca().twinx()
    ax2.plot(kmc_voltage_list, current_list, marker='o', markersize=5, linestyle='-', color='red', label='Log Scale')
    ax2.set_yscale('log')  # Set the right y-axis to log scale
    ax2.set_ylim(0.001, 10)

    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('Current (uA)')
    plt.title('Current vs applied voltage')
    plt.tight_layout()
    plt.savefig('I-V.jpg')

if __name__ == '__main__':
    main()
