import sys
import numpy as np
import matplotlib.pyplot as plt

# plots current vs time using the values found in the output file
def plot_It(temperature, current, power, times, V):
    if len(temperature) < len(times):
        times.pop()

    if len(temperature) > len(times):
        temperature.pop()

    if len(current) < len(times):
        times.pop()
    
    if len(current) > len(times):
        current.pop()

    if len(power) < len(times):
        times.pop()

    if len(power) > len(times):
        power.pop()

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), tight_layout=True)  # Create a figure with two subplots

    axes[2].plot(times, temperature, marker='.', markersize=4)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Temperature ')
    axes[0].set_title('Applied Voltage = ' + V + ' V')

    axes[0].plot(times, current, marker='.', markersize=4)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Current ')

    axes[1].plot(times, power, marker='.', markersize=4)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Power (nW)')

    plt.suptitle('Temperature and Current vs Time')  # Add a common title
    plt.savefig('Temperature_Current_Power_'+str(V)+'.jpg', dpi=100)

def main():
    if len(sys.argv) != 2:
        print("Missing output filename! Correct usage: $python3 plot_current.py output.log")
        sys.exit()

    outfile = sys.argv[1]
    temperature = []
    current = []
    power = []
    V = 'V'
    times = []

    with open(outfile, "rt") as myfile:
        for line in myfile:
            # new applied voltage found
            if 'Applied' in line.split():
                V = line.split()[-2]
            if 'Current' in line.split():
                curr = float(line.split()[-1])
                current.append(curr)
            if 'temperature' in line.split():
                temp = float(line.split()[-1])
                temperature.append(temp)
            if 'power:' in line.split():
                powerTemp = (1e9)*float(line.split()[-1])
                power.append(powerTemp)
            if 'KMC' in line.split() and 'time' in line.split() and 'reached' not in line.split():
                times.append(float(line.split()[-1]))

    # plot the data for the last voltage
    plot_It(temperature, current, power, times, V)

if __name__ == '__main__':
    main()
