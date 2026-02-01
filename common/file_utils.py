"""Utility functions for filesystem operations."""

import os
import logging
import glob


def get_file_paths_from_dir(
    directory: str, extension: str, file_ids: set[str]
) -> set[str]:
    """Get file paths from a directory matching given file IDs and extension."""

    file_paths = set()
    search_pattern = os.path.join(directory, f"*.{extension.lstrip('.')}")
    for file_path in glob.glob(search_pattern):
        file_name = os.path.basename(file_path)

        if any(file_id in file_name for file_id in file_ids):
            file_paths.add(file_path)

    logging.info(
        f"Found {len(file_paths)} files with extension {extension} in {directory}."
    )
    return file_paths
