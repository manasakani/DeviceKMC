import re
import matplotlib.pyplot as plt

def extract_timings(file_path):
    timings = {'potential': [], 'power': []}

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        potential_match = re.search(r'Z - calculation time - potential from boundaries \[s\]: (\d+\.\d+)', line)
        power_match = re.search(r'Z - calculation time - dissipated power \[s\]: (\d+\.\d+)', line)

        if potential_match:
            timings['potential'].append(float(potential_match.group(1)))
        elif power_match:
            timings['power'].append(float(power_match.group(1)))

    return timings

def plot_box_plot(timings_1d, timings_2d, title):
    labels = ['1D potential', '2D potential', '1D power', '2D power']
    data = [timings_1d['potential'], timings_2d['potential'], timings_1d['power'], timings_2d['power']]

    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.ylabel('Time [s]')
    plt.savefig('2D_dist.png')

if __name__ == "__main__":
    file_1d_path = 'output_1D.txt'
    file_2d_path = 'output_2D.txt'

    timings_1d = extract_timings(file_1d_path)
    timings_2d = extract_timings(file_2d_path)

    plot_box_plot(timings_1d, timings_2d, 'Sparse matrix population: 1D (rowwise) vs 2D (elementwise)')
