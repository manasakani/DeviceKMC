import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


import seaborn

# parses data from xyz file
def read_xyz(filename):

    atoms = []
    coords = []
    lattice = []

    with open(filename, "rt") as myfile:
        for line in myfile:
            if len(line.split()) == 1:
                pass
            elif len(line.split()) == 0:
                pass
            elif line.split()[0] in ['d', 'Ti', 'N', 'Hf', 'O', 'Od']:
                pass
            # elif float(line.split()[1]) > 56.6351-5:
            #     pass
            elif line.split()[0] in ['Cell:', 'cell:']:
                lattice = line.split()[1:4]
            elif len(line.split()) == 6:
                atoms.append(line.split()[0])
                coords.append(line.split()[1:4])
            else:
                pass

    coords = np.asarray(coords, dtype=np.float64)

    return np.array(atoms), coords

# makes a scatter plot of the device atomic structure, highlighting vacancies and ions
def make_image(names, positions, potential, temperature, structure_folder, imname):

    x = [pos[0] for pos in positions]
    y = [pos[1] for pos in positions]
    
    colors = []
    for ind, element in enumerate(names):
        if element == 'V' or ind == 0:
            colors.append(-1)
        elif element == 'Od' or ind == len(names)-1:
            colors.append(+1)
        else:
            colors.append(0.0)

    colors = [color+1 for color in colors]
    reversed = False
    #fig = plt.figure()
    fig = plt.figure(figsize=(5, 6), tight_layout=True)

    ax = fig.add_subplot(3, 1, 1)
    ax.scatter(x, y, c=colors, s=2, alpha=0.5, cmap='viridis_r')
    # ax.scatter(x, y, c=colors, s=2, alpha=0.5, cmap='coolwarm')
    # ax.scatter(x, y, c=blue, s=2, alpha=0.5)
    ax.grid(False)
    ax.get_xaxis().set_ticks([])

    ax = fig.add_subplot(3, 1, 2)
    ax.grid(True)
    ax.scatter(x, potential, c=y, s=2, alpha=0.5, cmap='coolwarm')
    ax.set_ylim(ymin=np.min(potential)-2, ymax = np.max(potential)+2)
    ax.set_xlabel("x position(s) (A)")
    ax.set_ylabel("Potential (V)")
    
    ax = fig.add_subplot(3, 1, 3)
    ax.grid(True)
    ax.scatter(x, temperature, c=y, s=2, alpha=0.5, cmap='coolwarm')
    #ax.set_ylim(ymin=250)#, ymax = 3000)
    ax.set_xlabel("x position(s) (A)")
    ax.set_ylabel("Dissipated Power (W)")

    plt.savefig(structure_folder+'/'+imname)


# iterates over input directories and makes all the images
def main():
	
    structure_xyz = '/usr/scratch/attelas8/manasa/DeviceKMC/tests/7-crossbar_selected/Results_8.000000/snapshot_63000.xyz'
    imname = structure_xyz[0:-4]+'.jpg'
       
    names, positions = read_xyz(structure_xyz)
    print(positions)
    x = [pos[0]/10 for pos in positions]
    y = [pos[1]/10 for pos in positions]
    z = [pos[2]/10 for pos in positions]
    plt.style.use('seaborn') 
    seaborn.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
    
    # plt.scatter(y, z, c=x, cmap="GnBu", s=100, alpha=0.4, edgecolor='none', linewidth=1)
    # plt.scatter(y, z, c=x, cmap="twilight", s=100, alpha=0.3, edgecolor='black', linewidth=0.3)
    plt.scatter(y, z, s=100, alpha=0.1, edgecolor='black', linewidth=0.3)
    
    cbar = plt.colorbar(label='x (nm)')  # Add label to colorbar
    cbar.ax.set_yticks([])
    cbar.ax.set_yticklabels([])

    # contacts:

    # top contact:
    x_start1, x_end1 = 55.297001/10, 79.051399/10
    y_start1, y_end1 =  0.000000/10, 77.691803/10

    x_start3, x_end3 = 1.467400/10, 26.139299/10
    y_start3, y_end3 = 0.000000/10, 77.691803/10

    # bottom contact:
    x_start2, x_end2 = 1.467390/10, 79.968903/10
    y_start2, y_end2 = 0.0/10, 24.417400/10

    x_start4, x_end4 = 1.467390/10, 79.968903/10
    y_start4, y_end4 = 53.274399/10, 77.691803/10

    rect1 = patches.Rectangle((x_start1, y_start1), x_end1 - x_start1, y_end1 - y_start1, linewidth=2, edgecolor='blue', facecolor='none', alpha=0.5)
    rect2 = patches.Rectangle((x_start2, y_start2), x_end2 - x_start2, y_end2 - y_start2, linewidth=2, edgecolor='green', facecolor='none', alpha=0.5)
    
    rect3 = patches.Rectangle((x_start3, y_start3), x_end3 - x_start3, y_end3 - y_start3, linewidth=2, edgecolor='black', facecolor='none', alpha=0.5)
    rect4 = patches.Rectangle((x_start4, y_start4), x_end4 - x_start4, y_end4 - y_start4, linewidth=2, edgecolor='black', facecolor='none', alpha=0.5)

    # Add rectangles to the plot
    plt.gca().add_patch(rect1)
    plt.gca().add_patch(rect2)
    plt.gca().add_patch(rect3)
    plt.gca().add_patch(rect4)

    # Create legend for rectangles
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, alpha=0.5, label='selected top contact'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, alpha=0.5, label='selected bottom contact')]

    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.1))



    plt.xlabel('y (nm)')
    plt.ylabel('z (nm)')
    plt.axis('square')
    plt.savefig('figure.png')
    
    
    #make_image(names, coords, potential, temperature, structure_folder, imname)


if __name__ == '__main__':
    main()
