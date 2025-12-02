#%% imports
import os
from pathlib import Path
import numpy as np

from scripts.helper.features_files import (
    features_file_valid as fet_val,
    load_features_from_npz,
)
from scripts.pca import perform_pca

#%% Train features extraction helpers
def ensure_train_features_npz(
    transformer,
    name: str,
    dataset_dir,
    train_df,
    features_output_dir,
    npz_filename: str,
    force_reextract: bool,
    split: str = "train",
):
    features_output_dir = Path(features_output_dir)
    features_output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = os.path.join(features_output_dir, npz_filename)

    if force_reextract or not fet_val(npz_path, train_df):
        print(f"[{name}] Extracting {split} features...")
        transformer.process_dataset(
            dataset_dir=dataset_dir,
            photos_output_dir=None,
            features_output_dir=str(features_output_dir),
            save_features=True,
            save_visualizations=False,
            split=split,
        )
    else:
        print(f"[{name}] Using existing {split} features.")

    return npz_path


def ensure_test_features_npz(
    transformer,
    name: str,
    dataset_dir,
    test_df,
    features_output_dir,
    npz_filename: str,
    force_reextract: bool,
    splits=("test", "valid"),
):

    features_output_dir = Path(features_output_dir)
    features_output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = features_output_dir / npz_filename

    if force_reextract or not fet_val(npz_path, test_df):
        print(f"[{name}] Extracting test features from splits: {splits} ...")

        all_features = {}
        for split in splits:
            feats_split = transformer.process_dataset(
                dataset_dir=dataset_dir,
                photos_output_dir=None,
                features_output_dir=None,
                save_features=False,
                save_visualizations=False,
                split=split,
            )
            all_features.update(feats_split)

        np.savez(npz_path, **all_features)
        print(f"[{name}] Saved combined test features to {npz_path}")
    else:
        print(f"[{name}] Using existing test features.")

    return npz_path


def compute_pca_for_feature(
    name: str,
    train_npz_path,
    test_npz_path,
    train_df,
    test_df,
):
    print(f"\n=== PCA: {name} ===")
    X_train = load_features_from_npz(train_npz_path, train_df)
    X_test = load_features_from_npz(test_npz_path, test_df)
    print(f"[{name}] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # keep your original order: (X_test, X_train)
    X_train_pca, X_test_pca = perform_pca(X_test, X_train)
    return X_train_pca, X_test_pca
