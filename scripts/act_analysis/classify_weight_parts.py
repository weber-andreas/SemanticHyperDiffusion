import os
import sys
import argparse
import logging
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline, Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_classifier(model_name: str, random_state: int = 42) -> Any:
    """Factory method to return a sklearn classifier pipeline with optimized defaults."""
    
    if model_name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(256, 256),
                activation='relu',
                solver='lbfgs',      # L-BFGS is better here than Adam
                alpha=0.01,        
                max_iter=2000,
                random_state=random_state
            )
        )
    elif model_name == "svm":
        # The High-Dim Specialist
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel='linear', 
                C=1.0,           
                gamma='scale',       # Auto-scales based on feature variance
                class_weight='balanced',
                cache_size=1000,
                random_state=random_state
            )
        )
    elif model_name == "rf":
        return make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=500,    # Best in grid search
                max_features='sqrt', 
                min_samples_split=5,
                n_jobs=-1,
                random_state=random_state
            )
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def run_grid_search(model_name: str, X_train, y_train, random_state: int = 42) -> Any:
    """Performs exhaustive search for best hyperparameters."""
    logger.info(f"Running Grid Search for {model_name.upper()}...")
    
    base_clf = None
    param_grid = {}
    
    if model_name == "mlp":
        base_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(max_iter=1000, random_state=random_state))
        ])
        param_grid = {
            'mlp__solver': ['lbfgs', 'adam'],
            'mlp__alpha': [0.0001, 0.01, 0.1],
            'mlp__hidden_layer_sizes': [(1024, 256), (512, 128), (256,)]
        }
    elif model_name == "svm":
        base_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=random_state))
        ])
        param_grid = {
            'svm__C': [1.0, 10.0, 100.0],
            'svm__kernel': ['rbf', 'linear', 'poly'],
            'svm__gamma': ['scale', 'auto']
        }
    elif model_name == "rf":
        base_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=random_state, n_jobs=-1))
        ])
        param_grid = {
            'rf__n_estimators': [100, 500],
            'rf__max_depth': [None, 20, 50],
            'rf__min_samples_split': [2, 5]
        }
        
    grid = GridSearchCV(base_clf, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    
    logger.info(f"Best Parameters: {grid.best_params_}")
    logger.info(f"Best CV Score: {grid.best_score_:.4f}")
    
    return grid.best_estimator_


def extract_part_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Parses a state_dict and flattens weights per semantic part."""
    parts_data = {}
    
    # Group keys by part name
    keys_by_part = {}
    for key in state_dict.keys():
        if not key.startswith("parts."):
            continue
        
        # parts.wing.net.0.weight -> part_name = wing
        segments = key.split('.')
        part_name = segments[1]
        
        if part_name not in keys_by_part:
            keys_by_part[part_name] = []
        keys_by_part[part_name].append(key)

    # Flatten weights for each part
    for part_name, keys in keys_by_part.items():
        sorted_keys = sorted(keys)
        weights = []
        for k in sorted_keys:
            weights.append(state_dict[k].cpu().flatten().numpy())
            
        if weights:
            parts_data[part_name] = np.concatenate(weights)
            
    return parts_data

def load_dataset(mlp_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: # Changed return type
    """Loads all checkpoints and creates dataset (X, y_strings, ids)."""
    logger.info(f"Loading weights from {mlp_dir}...")
    
    X_list = []
    y_list = []
    id_list = []
    
    files = [f for f in os.listdir(mlp_dir) if f.endswith('.pth')]
    if not files:
        raise FileNotFoundError(f"No .pth files found in {mlp_dir}")

    for fname in files:
        path = os.path.join(mlp_dir, fname)
        
        # occ_1a2b3c_model.pth -> 1a2b3c
        try:
            parts = fname.split('_')
            shape_id = next((p for p in parts if len(p) > 8 and any(char.isdigit() for char in p)), parts[1])
        except:
            shape_id = fname

        try:
            state_dict = torch.load(path, map_location='cpu')
            parts_weights = extract_part_weights(state_dict)
            
            for part_name, weight_vector in parts_weights.items():
                X_list.append(weight_vector)
                y_list.append(part_name)
                id_list.append(shape_id) # Track outlier Ids
                
        except Exception as e:
            logger.warning(f"Failed to process {fname}: {e}")

    if not X_list:
        raise ValueError("No valid weight samples found.")

    X = np.array(X_list)
    y = np.array(y_list)
    ids = np.array(id_list)
    
    logger.info(f"Loaded {len(X)} samples.")
    logger.info(f"Feature dimension (weight vector size): {X.shape[1]}")
    
    return X, y, ids

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Weight Space Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Confusion matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate semantic decomposition via weight classification.")
    parser.add_argument('--mlp_dir', type=str, required=True, 
                        help="Directory containing overfitted MLP checkpoints (.pth)")
    parser.add_argument('--model', type=str, default='svm', choices=['mlp', 'svm', 'rf'],
                        help="Type of classifier to use.")
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                        help="Directory to save plots and metrics.")
    parser.add_argument('--test_size', type=float, default=0.2,
                        help="Fraction of data to use for testing.")
    parser.add_argument('--grid_search', action='store_true',
                        help="If set, runs exhaustive Grid Search to find best params instead of using defaults.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    X, y_raw, ids = load_dataset(args.mlp_dir)
    
    # Encode Labels to Classes
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_
    logger.info(f"Classes found: {class_names}")

    # Data Split for testing
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y_encoded, ids, test_size=args.test_size, random_state=42, stratify=y_encoded
    )
    
    if args.grid_search:
        clf = run_grid_search(args.model, X_train, y_train)
    else:
        logger.info(f"Initializing {args.model.upper()} classifier with optimized defaults...")
        clf = get_classifier(args.model)
        logger.info("Training classifier...")
        clf.fit(X_train, y_train)
    
    logger.info("Evaluating...")
    y_pred = clf.predict(X_test)
    
    total_acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test Set Accuracy: {total_acc:.4f}")
    
    report_str = classification_report(y_test, y_pred, target_names=class_names)
    
    # Calculate Per-Class Accuracy = Per-Class Recall
    cm = confusion_matrix(y_test, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print(f"\n--- Model: {args.model.upper()} ---")
    print(f"Total Accuracy: {total_acc:.2%}")
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:<15}: {per_class_acc[i]:.2%}")
    print("\n" + report_str)
    
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Model: {args.model.upper()}\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Total Accuracy: {total_acc:.4f}\n\n")
        
        f.write("Per-Class Accuracy (True Positive Rate):\n")
        f.write("-" * 40 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<15}: {per_class_acc[i]:.4f}\n")
        f.write("-" * 40 + "\n\n")
        
        f.write("Full Classification Report:\n")
        f.write("-" * 40 + "\n")
        f.write(report_str)

    plot_confusion_matrix(y_test, y_pred, class_names, 
                          os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Save Misclassified IDs
    misclassified_path = os.path.join(args.output_dir, f'misclassified_ids_{args.model}.txt')
    logger.info(f"Saving unique misclassified shape IDs to {misclassified_path}...")
    
    bad_shapes = set()
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            bad_shapes.add(ids_test[i])
            
    with open(misclassified_path, 'w') as f:
        for sid in sorted(list(bad_shapes)):
            f.write(f"{sid}\n")

if __name__ == "__main__":
    main()