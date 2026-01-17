import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import sys

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_ids(filepath):
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return set()
    
    ids = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Shape_ID") or line.startswith("-"):
                continue
            
            # Handle table format or simple list
            raw_id = line.split('|')[0].strip()
            # Strip extensions if present
            clean_id = raw_id.replace('.obj', '').replace('.ply', '').replace('.off', '')
            ids.add(clean_id)
    return ids

def load_full_dataset(train_split_path):
    """Loads train, val, and test splits to get the total universe of shapes."""
    base_dir = os.path.dirname(train_split_path)
    if not base_dir: base_dir = "."
    
    splits = ['train_split.lst', 'val_split.lst', 'test_split.lst']
    total_ids = set()
    
    logger.info("Loading Ground Truth Dataset:")
    for split_name in splits:
        path = os.path.join(base_dir, split_name)
        if os.path.exists(path):
            ids = load_ids(path)
            total_ids.update(ids)
            logger.info(f"  - Loaded {len(ids)} shapes from {split_name}")
            
    return total_ids

def main():
    parser = argparse.ArgumentParser(description="Visualize semantic outlier distribution.")
    parser.add_argument('--files', nargs='+', required=True, help="List of misclassified ID files.")
    parser.add_argument('--train_split', type=str, required=True, help="Path to train_split.lst.")
    parser.add_argument('--output_dir', type=str, default='analysis_results')
    args = parser.parse_args()

    all_shapes = load_full_dataset(args.train_split)
    classifier_failures = [load_ids(f) for f in args.files]
    
    # Compute "Ambiguity Score" i.e how many classifiers fail
    failure_counts = Counter()
    for sid in all_shapes:
        failure_counts[sid] = 0
        
    for fail_set in classifier_failures:
        for sid in fail_set:
            if sid in failure_counts:
                failure_counts[sid] += 1
    
    scores = list(failure_counts.values())
    max_score = len(classifier_failures) 
    
    # Plot Histogram (Same Style as Loss outliers)
    plt.figure(figsize=(10, 6))
    
    # Centered bins
    bins = np.arange(max_score + 2) - 0.5
    
    n, bins, patches = plt.hist(scores, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, log=False)
    
    cutoff_x = max_score - 0.5
    plt.axvline(x=cutoff_x, color='red', linestyle='dashed', linewidth=2, label='Outlier Cutoff')
    
    plt.title(f"Distribution of Semantic Ambiguity (N={len(all_shapes)})")
    plt.xlabel("Number of Classifiers Failed")
    plt.ylabel("Frequency")
    plt.xticks(range(max_score + 1))
    plt.legend()
    plt.grid(axis='y', alpha=0.5)

    # text annotations for small bars
    for i in range(max_score + 1):
        if i < len(n):
            count = int(n[i])
            if count > 0:
                plt.text(i, count, f"{count}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Save Plots
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'semantic_outlier_hist.png')
    plt.savefig(output_path, bbox_inches='tight')
    logger.info(f"Histogram saved to: {output_path}")

    # Save ids failed by all classifiers as "problematic shapes"
    intersection_ids = [sid for sid, count in failure_counts.items() if count == max_score]
    list_path = os.path.join(args.output_dir, 'semantic_outliers_intersection.txt')
    with open(list_path, 'w') as f:
        for sid in sorted(intersection_ids):
            f.write(f"{sid}\n")
    logger.info(f"Intersection list ({len(intersection_ids)} shapes) saved to: {list_path}")

if __name__ == "__main__":
    main()