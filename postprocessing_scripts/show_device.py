import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_rgba

# parses data from xyz file
def read_xyz(filename):

    atoms = []
    coords = []
    potential = []
    temperature = []
    lattice = []

    with open(filename, "rt") as myfile:
        for line in myfile:
            if len(line.split()) == 1:
                pass
            elif len(line.split()) == 0:
                pass
            # elif line.split()[0] == 'd':
            #     pass
            elif line.split()[0] in ['Cell:', 'cell:']:
                lattice = line.split()[1:4]
            elif len(line.split()) == 6:
                atoms.append(line.split()[0])
                coords.append(line.split()[1:3])
                potential.append(line.split()[4])
                temperature.append(line.split()[5])
            else:
                pass

    coords = np.asarray(coords, dtype=np.float64)
    potential = np.asarray(potential, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    lattice = np.asarray(lattice, dtype=np.float64)

    return np.array(atoms), coords, potential, temperature

# makes a scatter plot of the device atomic structure, highlighting vacancies and ions
def make_image(names, positions, potential, temperature, structure_folder, imname):

    x = [pos[0] for pos in positions]
    y = [pos[1] for pos in positions]
    
    colors = []
    for ind, element in enumerate(names):
        if element == 'V' or ind == 0:
            r, g, b = to_rgb('red')
            colors.append((r, g, b, 0.8))
        elif element == 'Od' or ind == len(names)-1:
            r, g, b = to_rgb('blue')
            colors.append((r, g, b, 0.8))
        elif element in ['Ti', 'N', 'Hf', 'O']:
            r, g, b = to_rgb('gray')
            colors.append((r, g, b, 0.1))
        elif element in ['d']:
            r, g, b = to_rgb('gray')
            colors.append((r, g, b, 0.05))
        else:
            r, g, b = to_rgb('gray')
            colors.append((r, g, b, 0.0))

    reversed = False
    #fig = plt.figure()
    fig = plt.figure(figsize=(5, 6), tight_layout=True)

    ax = fig.add_subplot(3, 1, 1)
    ax.scatter(x, y, c=colors, s=2)
    ax.grid(False)
    ax.get_xaxis().set_ticks([])

    ax = fig.add_subplot(3, 1, 2)
    ax.grid(True)
    ax.scatter(x, potential, c=y, s=2, alpha=0.5, cmap='coolwarm')
    # ax.set_ylim(ymin=np.min(potential)-2, ymax = np.max(potential)+2)
    ax.set_xlabel("x position(s) (A)")
    ax.set_ylabel("Potential (V)")
    
    ax = fig.add_subplot(3, 1, 3)
    ax.grid(True)
    ax.scatter(x, temperature, c=y, s=2, alpha=0.5, cmap='coolwarm')
    ax.set_xlabel("x position(s) (A)")
    ax.set_ylabel("Temperature (K)")

    plt.savefig(structure_folder+'/'+imname)


# iterates over input directories and makes all the images
def main():

    if len(sys.argv) < 2:
        print("Missing folder! Correct usage: $python3 show_device.py Results_X Results_Y ...")
        sys.exit()
    
    Results_Folders = sys.argv[1:]
    for structure_folder in Results_Folders:    
        structure_xyzs = [os.path.basename(x) for x in glob.glob(structure_folder+'/snapshot_*.xyz')]
        
        for structure_xyz in structure_xyzs:
            imname = structure_xyz[0:-4]+'.jpg'
          
            if not os.path.isfile(structure_folder+'/'+imname):
                structure_file = structure_folder + '/' + structure_xyz
                names, coords, potential, temperature = read_xyz(structure_file)
                make_image(names, coords, potential, temperature, structure_folder, imname)
                print("Made device image for " + structure_folder + "/" + imname)


if __name__ == '__main__':
    main()
