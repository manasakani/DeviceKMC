import sys
import numpy as np
import matplotlib.pyplot as plt



def plot_power(power, times, V):
    if len(power) < len(times):
        times.pop();

    if len(power) > len(times):
        power.pop();

    fig = plt.figure(figsize=(3, 4), tight_layout=True)
    print(times, power)
    plt.plot(times, power, marker='.', markersize=4)
    plt.xscale('log')
    #plt.xlim(1e-14, 1e-10)
    plt.xlabel('Time (s)')
    plt.ylabel('Power (nW)')
    plt.title('Applied Voltage = ' + V + ' V')
    plt.savefig('PowerTotal_'+str(V)+'.jpg', dpi=100)

def main():

    if len(sys.argv) != 2:
        print("Missing output filename! Correct usage: $python3 plot_current.py output.log")
        sys.exit()
 
    outfile = sys.argv[1]
    power = []
    V = 'V'
    times = []

    with open(outfile, "rt") as myfile:
        for line in myfile:
            # new applied voltage found
            
            if 'Applied' in line.split():
                #print(line.split())
                # finish the old voltage:
                if V != 'V':
                    break
                    shift = times[len(times)-1];
                # start collectin data for the next voltage:
                V = line.split()[-2]
                #times = [0]
                #temperature = []
            #if 'Dissipated' in line.split() and 'reached' not in line.split():
            if 'power:' in line.split():
                powerTemp = float(line.split()[-1])
                power.append(powerTemp)

            if 'KMC' in line.split() and 'time' in line.split() and 'reached' not in line.split():
                times.append(float(line.split()[-1]))

    # plot the data for the last voltage
    plot_power(power, times, V)

if __name__ == '__main__':
    main()
