#%% Imports and constants
import pandas as pd

from scripts.helper.main_pipeline_helper import (
    ensure_train_features_npz,
    ensure_test_features_npz,
    compute_pca_for_feature,
)

from scripts.dataset.unpack_root_dataset import main_unpack
from scripts.dataset.create_splits import create_splits, convert_class_name_to_index
from scripts.scaler import rescale_all_images, INPUT_DIR
from scripts.hog.hog import HOGTransformer
from scripts.lbp.lbp import LBPTransformer
from scripts.orb.orb import ORBTransformer
from scripts.sift.sift import SIFTTransformer

from scripts.classifiers.svm import SVMClassifier
from scripts.classifiers.random_forests import RandomForestClassifier
# from scripts.classifiers.logreg_orv import OVRLogRegClassifier

from scripts.classifiers.metrics import calculate_metrics, plot_confusion_matrix

FORCE_REEXTRACT = True

#%% Extract dataset paths and labels from a given directory
main_unpack()
create_splits()

#%% Resize and pad images to a fixed size
rescale_all_images()

#%% Feature extractors
hog_transformer = HOGTransformer(visualize=False)
lbp_transformer = LBPTransformer(visualize=False)
orb_transformer = ORBTransformer(visualize=False)
sift_transformer = SIFTTransformer(visualize=False)

#%% Load data
data_train = pd.read_csv(f'{INPUT_DIR}/train.csv')
y_train = list(map(convert_class_name_to_index, data_train['class'].tolist()))

data_test = pd.read_csv(f'{INPUT_DIR}/test.csv')
y_test = list(map(convert_class_name_to_index, data_test['class'].tolist()))

#%% HOG train features
hog_train_npz = ensure_train_features_npz(
    transformer=hog_transformer,
    name="HOG",
    dataset_dir=INPUT_DIR,
    train_df=data_train,
    features_output_dir="output/hog/features/train",
    npz_filename="hog_features.npz",
    force_reextract=FORCE_REEXTRACT,
)

#%% LBP train features
lbp_train_npz = ensure_train_features_npz(
    transformer=lbp_transformer,
    name="LBP",
    dataset_dir=INPUT_DIR,
    train_df=data_train,
    features_output_dir="output/lbp/features/train",
    npz_filename="lbp_features.npz",
    force_reextract=FORCE_REEXTRACT,
)

#%% ORB train features
orb_train_npz = ensure_train_features_npz(
    transformer=orb_transformer,
    name="ORB",
    dataset_dir=INPUT_DIR,
    train_df=data_train,
    features_output_dir="output/orb/features/train",
    npz_filename="orb_features.npz",
    force_reextract=FORCE_REEXTRACT,
)

#%% SIFT train features
sift_train_npz = ensure_train_features_npz(
    transformer=sift_transformer,
    name="SIFT",
    dataset_dir=INPUT_DIR,
    train_df=data_train,
    features_output_dir="output/sift/features/train",
    npz_filename="sift_features.npz",
    force_reextract=FORCE_REEXTRACT,
)

#%% HOG test features
hog_test_npz = ensure_test_features_npz(
    transformer=hog_transformer,
    name="HOG",
    dataset_dir=INPUT_DIR,
    test_df=data_test,
    features_output_dir="output/hog/features/test",
    npz_filename="hog_features.npz",
    force_reextract=FORCE_REEXTRACT,
)

#%% LBP test features
lbp_test_npz = ensure_test_features_npz(
    transformer=lbp_transformer,
    name="LBP",
    dataset_dir=INPUT_DIR,
    test_df=data_test,
    features_output_dir="output/lbp/features/test",
    npz_filename="lbp_features.npz",
    force_reextract=FORCE_REEXTRACT,
)

#%% ORB test features (test + valid -> one npz)
orb_test_npz = ensure_test_features_npz(
    transformer=orb_transformer,
    name="ORB",
    dataset_dir=INPUT_DIR,
    test_df=data_test,
    features_output_dir="output/orb/features/test",
    npz_filename="orb_features.npz",
    force_reextract=FORCE_REEXTRACT,
)

#%% SIFT test features (test + valid -> one npz)
sift_test_npz = ensure_test_features_npz(
    transformer=sift_transformer,
    name="SIFT",
    dataset_dir=INPUT_DIR,
    test_df=data_test,
    features_output_dir="output/sift/features/test",
    npz_filename="sift_features.npz",
    force_reextract=FORCE_REEXTRACT,
)

#%% PCA
x_hog_train, x_hog_test = compute_pca_for_feature(
    name="HOG",
    train_npz_path=hog_train_npz,
    test_npz_path=hog_test_npz,
    train_df=data_train,
    test_df=data_test,
)

x_lbp_train, x_lbp_test = compute_pca_for_feature(
    name="LBP",
    train_npz_path=lbp_train_npz,
    test_npz_path=lbp_test_npz,
    train_df=data_train,
    test_df=data_test,
)

x_orb_train, x_orb_test = compute_pca_for_feature(
    name="ORB",
    train_npz_path=orb_train_npz,
    test_npz_path=orb_test_npz,
    train_df=data_train,
    test_df=data_test,
)

x_sift_train, x_sift_test = compute_pca_for_feature(
    name="SIFT",
    train_npz_path=sift_train_npz,
    test_npz_path=sift_test_npz,
    train_df=data_train,
    test_df=data_test,
)

#%% Classification - SVM
# HOG + SVM
svm_hog_classifier = SVMClassifier()
svm_hog_classifier.fit(x_hog_train, y_train)
y_hog_svm_pred = svm_hog_classifier.predict(x_hog_test)

# LBP + SVM
svm_lbp_classifier = SVMClassifier()
svm_lbp_classifier.fit(x_lbp_train, y_train)
y_lbp_svm_pred = svm_lbp_classifier.predict(x_lbp_test)

# ORB + SVM
svm_orb_classifier = SVMClassifier()
svm_orb_classifier.fit(x_orb_train, y_train)
y_orb_svm_pred = svm_orb_classifier.predict(x_orb_test)

# SIFT + SVM
svm_sift_classifier = SVMClassifier()
svm_sift_classifier.fit(x_sift_train, y_train)
y_sift_svm_pred = svm_sift_classifier.predict(x_sift_test)

#%% Classification - Random Forest
# HOG + RF
rf_hog_classifier = RandomForestClassifier()
rf_hog_classifier.fit(x_hog_train, y_train)
y_hog_rf_pred = rf_hog_classifier.predict(x_hog_test)

# LBP + RF
rf_lbp_classifier = RandomForestClassifier()
rf_lbp_classifier.fit(x_lbp_train, y_train)
y_lbp_rf_pred = rf_lbp_classifier.predict(x_lbp_test)

# ORB + RF
rf_orb_classifier = RandomForestClassifier()
rf_orb_classifier.fit(x_orb_train, y_train)
y_orb_rf_pred = rf_orb_classifier.predict(x_orb_test)

# SIFT + RF
rf_sift_classifier = RandomForestClassifier()
rf_sift_classifier.fit(x_sift_train, y_train)
y_sift_rf_pred = rf_sift_classifier.predict(x_sift_test)

# #%% Classification - OVR Logistic Regression
#
# # HOG + LogReg OVR
# logreg_hog_classifier = OVRLogRegClassifier()
# logreg_hog_classifier.fit(x_hog_train, y_train)
# y_hog_logreg_pred = logreg_hog_classifier.predict(x_hog_test)
#
# # LBP + LogReg OVR
# logreg_lbp_classifier = OVRLogRegClassifier()
# logreg_lbp_classifier.fit(x_lbp_train, y_train)
# y_lbp_logreg_pred = logreg_lbp_classifier.predict(x_lbp_test)
#
# # ORB + LogReg OVR
# logreg_orb_classifier = OVRLogRegClassifier()
# logreg_orb_classifier.fit(x_orb_train, y_train)
# y_orb_logreg_pred = logreg_orb_classifier.predict(x_orb_test)
#
# # SIFT + LogReg OVR
# logreg_sift_classifier = OVRLogRegClassifier()
# logreg_sift_classifier.fit(x_sift_train, y_train)
# y_sift_logreg_pred = logreg_sift_classifier.predict(x_sift_test)

#%% Metrics 
PLOT_CONF_MATRICES = True

results = {
    # SVM
    ("HOG",  "SVM"): y_hog_svm_pred,
    ("LBP",  "SVM"): y_lbp_svm_pred,
    ("ORB",  "SVM"): y_orb_svm_pred,
    ("SIFT", "SVM"): y_sift_svm_pred,

    # Random Forest
    ("HOG",  "RF"): y_hog_rf_pred,
    ("LBP",  "RF"): y_lbp_rf_pred,
    ("ORB",  "RF"): y_orb_rf_pred,
    ("SIFT", "RF"): y_sift_rf_pred,

    # # OVR Logistic Regression
    # ("HOG",  "LogReg-OVR"): y_hog_logreg_pred,
    # ("LBP",  "LogReg-OVR"): y_lbp_logreg_pred,
    # ("ORB",  "LogReg-OVR"): y_orb_logreg_pred,
    # ("SIFT", "LogReg-OVR"): y_sift_logreg_pred,
}

for (feature_name, clf_name), y_pred in results.items():
    print("\n" + "=" * 60)
    print(f"Metrics for {feature_name} + {clf_name}")
    print("=" * 60)

    accuracy, precision, recall, specificity, f1, balanced_acc = calculate_metrics(y_test, y_pred)

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro)   : {recall:.4f}")
    print(f"Specificity      : {specificity:.4f}")
    print(f"F1-score (macro) : {f1:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")

    if PLOT_CONF_MATRICES:
        print("Plotting confusion matrix...")
        plot_confusion_matrix(y_test, y_pred)
