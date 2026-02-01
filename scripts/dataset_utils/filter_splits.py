import argparse
import os
import sys
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_ids(filepath):
    """Parses a file to extract a set of cleaned shape IDs."""
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return set()
    
    ids = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, headers, or comments
            if not line or line.startswith("Shape_ID") or line.startswith("-"):
                continue
            
            # Handle table format or simple list
            raw_id = line.split('|')[0].strip()
            # Strip extensions if present
            clean_id = raw_id.replace('.obj', '').replace('.ply', '').replace('.off', '')
            ids.add(clean_id)
    return ids

def save_split_overwrite(filepath, id_set):
    """Overwrites the file with the filtered set of IDs, sorted alphabetically."""
    try:
        with open(filepath, 'w') as f:
            for sid in sorted(list(id_set)):
                f.write(f"{sid}\n")
        return True
    except Exception as e:
        logger.error(f"Failed to write to {filepath}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Filter dataset splits by removing specific IDs (OVERWRITES FILES).")
    parser.add_argument('--files', nargs='+', required=True, help="List of files containing IDs to remove.")
    parser.add_argument('--train_split', type=str, required=True, help="Path to train_split.lst (determines directory of other splits).")
    args = parser.parse_args()

    # 1. Load IDs to exclude
    ids_to_remove = set()
    logger.info("Loading IDs to exclude...")
    for file_path in args.files:
        loaded = load_ids(file_path)
        ids_to_remove.update(loaded)
        logger.info(f"  - Loaded {len(loaded)} IDs from {os.path.basename(file_path)}")
    
    logger.info(f"Total unique IDs to remove: {len(ids_to_remove)}")

    # 2. Process and Overwrite Splits
    base_dir = os.path.dirname(args.train_split)
    if not base_dir: base_dir = "."
    
    splits = ['train_split.lst', 'val_split.lst', 'test_split.lst']

    logger.info("Processing and overwriting splits...")
    
    for split_name in splits:
        target_path = os.path.join(base_dir, split_name)
        
        if os.path.exists(target_path):
            # Load original IDs
            original_ids = load_ids(target_path)
            
            # Filter
            filtered_ids = original_ids - ids_to_remove
            
            # Calculate stats
            removed_count = len(original_ids) - len(filtered_ids)
            
            if removed_count > 0:
                # Overwrite the file
                success = save_split_overwrite(target_path, filtered_ids)
                if success:
                    logger.info(f"[{split_name}] OVERWRITTEN. Removed {removed_count} IDs. New count: {len(filtered_ids)}")
            else:
                logger.info(f"[{split_name}] No IDs to remove. File touched but content unchanged.")
                # We save anyway to ensure format consistency (sorting/cleaning), 
                # or you can skip this if you want to preserve exact file state.
                save_split_overwrite(target_path, filtered_ids)

        else:
            logger.warning(f"Split file not found, skipping: {target_path}")

if __name__ == "__main__":
    main()