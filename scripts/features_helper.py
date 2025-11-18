from pathlib import Path
import numpy as np


def load_features_from_npz(npz_path, data_frame, image_col="image_path"):
    """
    Load feature vectors from a .npz file.

    This assumes that:
    - the .npz file contains one vector per image,
    - keys in the .npz file match values in the `image_col` column
      (e.g. "class_name/image_name.jpg").
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Feature file not found: {npz_path}")

    feats = np.load(npz_path)
    X = []

    for _, row in data_frame.iterrows():
        raw_key = str(row[image_col])
        norm_key = raw_key.replace("\\", "/")

        if norm_key in feats:
            X.append(feats[norm_key])
        elif raw_key in feats:
            X.append(feats[raw_key])
        else:
            raise KeyError(f"Missing feature vector for key: {raw_key} (normalized: {norm_key})")

    return np.vstack(X)


def features_file_valid(npz_path, data_frame, image_col="image_path") -> bool:
    """
    Check if a .npz feature file exists and contains feature vectors
    for all images listed in `data_frame[image_col]`.

    This function DOES NOT load all feature vectors into memory.
    It only:
      - checks if the file can be opened,
      - checks if all expected keys are present.
    """
    npz_path = Path(npz_path)

    if not npz_path.exists():
        print(f"[features] File does not exist: {npz_path}")
        return False

    try:
        feats = np.load(npz_path)
    except Exception as e:
        print(f"[features] Failed to open {npz_path}: {e}")
        return False

    try:
        for _, row in data_frame.iterrows():
            raw_key = str(row[image_col])
            norm_key = raw_key.replace("\\", "/")

            if norm_key in feats:
                continue
            if raw_key in feats:
                continue

            print(f"[features] Missing key in npz: {raw_key} (normalized: {norm_key})")
            return False
    finally:
        if hasattr(feats, "close"):
            feats.close()

    return True