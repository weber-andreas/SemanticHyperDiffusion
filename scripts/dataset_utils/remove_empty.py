""" "This simple script removes empty files from a specified directory."""

import os


def remove_empty_files(directory):
    """Removes empty files from the specified directory.

    Args:
        directory (str): The path to the directory to clean up.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
            os.remove(file_path)
            print(f"Removed empty file: {file_path}")


if __name__ == "__main__":
    target_directory = "chair"  # Replace with your target directory
    remove_empty_files(target_directory)
