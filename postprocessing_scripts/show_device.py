import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

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
            elif line.split()[0] == 'd':
                pass
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
    #ax.scatter(x, y, c=colors, s=2, alpha=0.5, cmap='viridis_r')
    ax.scatter(x, y, c=colors, s=2, alpha=0.5, cmap='coolwarm')
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
    ax.scatter(x, temperature*1e9, c=y, s=2, alpha=0.5, cmap='coolwarm')
    ax.set_ylim(ymin=0, ymax=50)#, ymax = 3000)
    ax.set_xlabel("x position(s) (A)")
    ax.set_ylabel("Dissipated Power (nW)")

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
