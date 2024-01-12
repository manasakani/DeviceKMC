import matplotlib.pyplot as plt
import sys

def main():
    if len(sys.argv) != 2:
        print("Missing output filename! Correct usage: $python3 plot_temperature.py output.log")
        sys.exit()

    outfile = sys.argv[1]

    print("reading from: " + outfile)
    
    # Initialize lists to store KMC time and current values
    kmc_time_list = [0.0]
    temperature_list = [0.0]

    reset_time = 0.0
    kmc_time = 0.0

    # Read the 'output.txt' file
    with open(outfile, "rt") as myfile:

        # Iterate through each line in the file
        for line in myfile:

            if "Applied Voltage =" in line:
                reset_time = kmc_time_list[-1]

            if "KMC time is:" in line:
                kmc_time = float(line.split(":")[-1])
                kmc_time_list.append(kmc_time + reset_time)

            if "temperature [K]" in line:
                temperature = float(line.split()[-1]) 
                temperature_list.append(temperature)
    
    plt.figure(figsize=(5, 4))

    # Plotting the data on the left y-axis (linear scale)
    plt.plot(kmc_time_list, temperature_list, marker='o', markersize=1, linestyle='-', label='Linear Scale')
    plt.yscale('linear')  # Set the left y-axis to linear scale
    plt.ylim(100, 1000)

    # Creating a secondary y-axis with log scale
    ax2 = plt.gca().twinx()
    ax2.plot(kmc_time_list, temperature_list, marker='o', markersize=1, linestyle='-', color='red', label='Log Scale')
    ax2.set_yscale('log')  # Set the right y-axis to log scale
    ax2.set_ylim(100, 1000)

    plt.xlabel('KMC Time')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs KMC Time')
    plt.tight_layout()
    plt.savefig('Temperature.jpg')

if __name__ == '__main__':
    main()
