import os
import filecmp

# List of test folders
TEST_FOLDERS = [
    "1-potential",
    "2-globaltemp",
    "3-localtemp"
]

def compare_results(test_folder):
    compare_folder = os.path.join(test_folder, "compare")

    if not os.path.exists(compare_folder):
        print(f"Test '{test_folder}' skipped. Missing compare folder.")
        return

    for result_folder in os.listdir(test_folder):
        if result_folder.startswith("Results_"):
            result_path = os.path.join(test_folder, result_folder)
            compare_path = os.path.join(compare_folder, result_folder)

            if os.path.isdir(result_path) and os.path.isdir(compare_path):
                comparison = filecmp.dircmp(result_path, compare_path)

                if not comparison.diff_files:
                    print(f"Test '{test_folder}/{result_folder}' PASSED. Results_X matches compare/{result_folder}.")
                else:
                    print(f"Test '{test_folder}/{result_folder}' FAILED. Differences found:")
                    for file in comparison.diff_files:
                        print(f"   - {file}")

if __name__ == "__main__":
    for test_folder in TEST_FOLDERS:
        compare_results(test_folder)
