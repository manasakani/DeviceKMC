# This script contains some functionality to re-order the xyz file such that the sites 
# we want to treat as boundary conditions for the applied potential are at the beginning/end. 

import numpy as np

# Modify this function to sort coordinates by x-coordinate
def write_sorted_xyz(file_path, atoms, coordinates):
    with open(file_path, 'a') as xyz_file:
        for atom, coord in zip(atoms, coordinates):
            line = "{} {:.6f} {:.6f} {:.6f}\n".format(atom, coord[0], coord[1], coord[2])
            xyz_file.write(line)

# Write XYZ coordinates to a file
def write_xyz(file_path, atoms, coordinates):
    with open(file_path, 'a') as xyz_file:
        for atom, coord in zip(atoms, coordinates):
            line = "{} {:.6f} {:.6f} {:.6f}\n".format(atom, coord[0], coord[1], coord[2])
            xyz_file.write(line)


# Sort coordinates by x-coordinate
def sort_coordinates(atoms, coordinates):
    return zip(*sorted(zip(atoms, coordinates), key=lambda x: x[1][0]))

# Re-order coordinates according to specified order
def reorder_coordinates(wordline1, wordline2, oxide, interstitials, bitline1, bitline2):
    ordered_atoms = []
    ordered_coordinates = []

    # Compile ordered coordinates and atoms
    for atoms, coords in [wordline1, wordline2, oxide, interstitials, bitline1, bitline2]:
        ordered_atoms.extend(atoms)
        ordered_coordinates.extend(coords)

    return ordered_atoms, ordered_coordinates

# Read xyz coordinates from a file:
def read_xyz(file_path):
    atoms = []
    coordinates = []

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into words
            words = line.split()

            # Check if the line has at least 4 words (atom and x, y, z coordinates)
            if len(words) >= 4:
                # The first word is the atom
                atom = words[0]

                # The remaining words are x, y, z coordinates
                x, y, z = map(float, words[1:4])

                # Append the atom and coordinates to the respective lists
                atoms.append(atom)
                coordinates.append([x, y, z])

    # Convert the coordinates list to a NumPy array
    coordinates_array = np.array(coordinates)

    return atoms, coordinates


file_path = "crossbar_40.xyz"
file_path_interstitials = "interstitials_crossbar_40.xyz"

left_contact_end = 0.0
right_contact_start = 51.0

atoms, coordinates = read_xyz(file_path)
print("# Atoms:", len(atoms))
print("Coordinates:", np.shape(coordinates))

# *** Extract all the coordinates and atoms ***

# the number of contact atoms is determined by the left contact
left_contact_coords = []
left_contact_atoms = []

num_contact = 0
for atom, coord in zip(atoms, coordinates):
    if atom in ['Ti', 'N']:
        if coord[0] < left_contact_end:
            # print(coord[0])
            left_contact_coords.append(coord)
            left_contact_atoms.append(atom)            
        num_contact += 1

print("Num contact left: ", num_contact)

num_contact = 0
right_contact_coords = []
right_contact_atoms = []
for atom, coord in zip(reversed(atoms), reversed(coordinates)):
    if atom in ['Ti', 'N']:
        if coord[0] > right_contact_start:
            right_contact_coords.append(coord)
            right_contact_atoms.append(atom)
        num_contact += 1

print("Num contact right: ", num_contact)

right_contact_coords.reverse()
right_contact_atoms.reverse()

# Get the oxide atoms:
oxide = []
for i in range(len(atoms)):
    if atoms[i] not in ['Ti', 'N']:
        oxide.append([atoms[i], coordinates[i]])


# *** Group the atoms into the bit/word lines ***

# left contact is split by y
wordline1 = []
wordline2 = []
wordline3 = []

split_y1 = 130
split_y2 = 270

for atom, coord in zip(left_contact_atoms, left_contact_coords):
    if coord[1] < split_y1:
        wordline1.append([atom, coord])
    elif split_y1 < coord[1] < split_y2:
        wordline2.append([atom, coord])
    else:
        wordline3.append([atom, coord])

print("wordline1 #atoms: ", len(wordline1))
print("wordline2 #atoms: ", len(wordline2))
print("wordline3 #atoms: ", len(wordline3))

# right contact is split by z
bitline1 = []
bitline2 = []
bitline3 = []

split_z1 = 130
split_z2 = 270

for atom, coord in zip(right_contact_atoms, right_contact_coords):
    if coord[2] < split_z1:
        bitline1.append([atom, coord])
    elif split_z1 < coord[2] < split_z2:
        bitline2.append([atom, coord])
    else:
        bitline3.append([atom, coord])

print("bitline1 #atoms: ", len(bitline1))
print("bitline2 #atoms: ", len(bitline2))
print("bitline3 #atoms: ", len(bitline3))

# *** Extract the interstitials:
interstitials = []
interstitial_atoms, interstitial_coords = read_xyz(file_path_interstitials)
for atom, coord in zip(interstitial_atoms, interstitial_coords):
    interstitials.append([atom, coord])

# assemble the full list of coordinates:
# stack the biased contacts at the start and the end so they get placed in the boundaries
# this biases the middle bitline and wordline
device = []
device.append(wordline2)
device.append(wordline1)
device.append(wordline3)
device.append(oxide)
device.append(interstitials)
device.append(bitline1)
device.append(bitline3)
device.append(bitline2)
device = [item for sublist in device for item in sublist]


# Extract the first and last 144 elements from the entire 'device' list
first_144 = device[:144]
last_144 = device[-144:]

# Sort the elements by x-coordinate while excluding the first and last 144 elements
print("sorting...")
sorted_device = sorted(device[144:-144], key=lambda x: x[1][0])
print("sorted")

# print(sorted_device)

# Re-build the xyz file
xyz_file_path = "reordered_crossbar_40.xyz"

print("writing to xyz file...")
with open(xyz_file_path, 'w') as xyz_file:
    # Write total number of atoms to the file
    total_atoms = len(device)
    xyz_file.write(str(total_atoms) + '\n\n')
    
    write_xyz(xyz_file_path, *zip(*first_144))
    write_xyz(xyz_file_path, *zip(*sorted_device))
    write_xyz(xyz_file_path, *zip(*last_144))