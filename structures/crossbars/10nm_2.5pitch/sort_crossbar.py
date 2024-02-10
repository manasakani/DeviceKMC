# This script contains some functionality to re-order the xyz file such that the sites 
# we want to treat as boundary conditions for the applied potential are at the beginning/end. 

import numpy as np

# Write XYZ coordinates to a file
def write_xyz(file_path, atoms, coordinates):
    with open(file_path, 'a') as xyz_file:
        for atom, coord in zip(atoms, coordinates):
            line = "{} {:.6f} {:.6f} {:.6f}\n".format(atom, coord[0], coord[1], coord[2])
            xyz_file.write(line)


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


file_path = "crossbar_10_2.5.xyz"
file_path_interstitials = "interstitials_crossbar_10_2.5.xyz"

atoms, coordinates = read_xyz(file_path)
print("# Atoms:", len(atoms))
print("Coordinates:", np.shape(coordinates))

# *** Extract all the coordinates and atoms ***

# the number of contact atoms is determined by the left contact
num_contact = 0
left_contact_coords = []
left_contact_atoms = []
for atom, coord in zip(atoms, coordinates):
    if atom in ['Ti', 'N']:
        left_contact_coords.append(coord)
        left_contact_atoms.append(atom)
        num_contact += 1
    if atom in ['Hf', 'O']:
        break

counter = 0
right_contact_coords = []
right_contact_atoms = []
for atom, coord in zip(reversed(atoms), reversed(coordinates)):
    if atom in ['Ti', 'N']:
        right_contact_coords.append(coord)
        right_contact_atoms.append(atom)
        counter += 1
    if atom in ['Hf', 'O'] or counter == num_contact:
        break

right_contact_coords.reverse()
right_contact_atoms.reverse()

print("Num contact: ", num_contact)

# Get the oxide atoms:
oxide = []
for i in range(num_contact, len(atoms)-num_contact):
    oxide.append([atoms[i], coordinates[i]])

# Use this to check how/where the bit/wordlines are split:
# print("Left contact coordinates:", np.shape(left_contact_coords))
# print("x values:", np.unique( [coord[0] for coord in left_contact_coords] ))
# print("y values:", np.sort(np.unique( [coord[1] for coord in left_contact_coords] )))
# print("z values:", np.sort(np.unique( [coord[2] for coord in left_contact_coords] )))

# *** Group the atoms into the bit/word lines ***

# left contact is split by y
wordline1 = []
wordline2 = []
split_y = 50

for atom, coord in zip(left_contact_atoms, left_contact_coords):
    if coord[1] < split_y:
        wordline1.append([atom, coord])
    else:
        wordline2.append([atom, coord])

print("wordline1 #atoms: ", len(wordline1))
print("wordline2 #atoms: ", len(wordline2))

# right contact is split by z
bitline1 = []
bitline2 = []
split_z = 50

for atom, coord in zip(right_contact_atoms, right_contact_coords):
    if coord[2] < split_z:
        bitline1.append([atom, coord])
    else:
        bitline2.append([atom, coord])

print("bitline1 #atoms: ", len(bitline1))
print("bitline2 #atoms: ", len(bitline2))

# *** Extract the interstitials:
interstitials = []
interstitial_atoms, interstitial_coords = read_xyz(file_path_interstitials)
for atom, coord in zip(interstitial_atoms, interstitial_coords):
    interstitials.append([atom, coord])

# Re-build the xyz file:
xyz_file_path = "reordered_crossbar_10_2.5.xyz"

with open(xyz_file_path, 'w') as xyz_file: 
    xyz_file.write(str(len(atoms) + len(interstitials)) + '\n\n')

write_xyz(xyz_file_path, *zip(*wordline1))
write_xyz(xyz_file_path, *zip(*wordline2))
write_xyz(xyz_file_path, *zip(*oxide))
write_xyz(xyz_file_path, *zip(*interstitials))
write_xyz(xyz_file_path, *zip(*bitline1))
write_xyz(xyz_file_path, *zip(*bitline2))