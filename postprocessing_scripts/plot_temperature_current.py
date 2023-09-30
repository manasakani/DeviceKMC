import sys
import numpy as np
import matplotlib.pyplot as plt

# plots current vs time using the values found in the output file,

def plot_It(temperature, times, V):
    #print(temperature)
    #currents = np.asarray(currents, dtype=np.float64)
    #times = np.asarray(times, dtype=np.float64)

    #if np.shape(currents) != np.shape(times): 
    #    times = times[0:-1]
    if len(temperature) < len(times):
        times.pop();

    if len(temperature) > len(times):
        temperature.pop();

    fig = plt.figure(figsize=(3, 4), tight_layout=True)
    print(times, temperature)
    plt.plot(times, temperature, marker='.', markersize=4)
    #plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature ')
    plt.title('Applied Voltage = ' + V + ' V')
    plt.savefig('Temperature_'+str(V)+'.jpg', dpi=100)

def plot_I(current, times, V):
    #print(temperature)
    #currents = np.asarray(currents, dtype=np.float64)
    #times = np.asarray(times, dtype=np.float64)

    #if np.shape(currents) != np.shape(times): 
    #    times = times[0:-1]
    if len(current) < len(times):
        times.pop();

    if len(current) > len(times):
        current.pop();

    fig = plt.figure(figsize=(3, 4), tight_layout=True)
    #print(times)
    plt.plot(times, current, marker='.', markersize=4)
    #plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Current ')
    plt.title('Applied Voltage = ' + V + ' V')
    plt.savefig('Current_'+str(V)+'.jpg', dpi=100)

def main():

    if len(sys.argv) != 2:
        print("Missing output filename! Correct usage: $python3 plot_current.py output.log")
        sys.exit()
 
    outfile = sys.argv[1]
    temperature = []
    current = []
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
            if 'Current' in line.split():
                curr = float(line.split()[-1])
                current.append(curr)
            if 'temperature' in line.split():
                temp = float(line.split()[-1])
                temperature.append(temp)

            if 'KMC' in line.split() and 'time' in line.split() and 'reached' not in line.split():
                times.append(float(line.split()[-1]))

    # plot the data for the last voltage
    plot_It(temperature, times, V)
    plot_I(current, times, V)

if __name__ == '__main__':
    main()
