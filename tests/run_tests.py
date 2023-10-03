import os
import subprocess

EXECUTABLE_PATH = "../../bin/runKMC"

TEST_FOLDERS = [
    "1-potential",
    "2-globaltemp",
    "3-localtemp"
]

def run_test(test_folder):

    input_file = os.path.join(test_folder, "parameters.txt")

    if not os.path.isfile(input_file):
        print(f"Input file '{input_file}' not found. Skipping test '{test_folder}'.")
        return

    print(f"Running test '{test_folder}'...")
    try:

        os.chdir(test_folder)
        result = subprocess.run([EXECUTABLE_PATH, input_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output_str = result.stdout.decode()
        error_str = result.stderr.decode()

        print("Test output:")
        print(result.stdout)
        print("Test errors:")
        print(result.stderr)

    except Exception as e:
        print(f"Error while running test '{test_folder}': {e}")

if __name__ == "__main__":
    for test_folder in TEST_FOLDERS:
        run_test(test_folder)

