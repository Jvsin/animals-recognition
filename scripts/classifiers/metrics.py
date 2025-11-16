#%% Imports 
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt 

#%% Calculate metrics - accuracy, sensitivity, specificity, precision, f1, balanced accuracy
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    accuracy = np.trace(cm) / np.sum(cm)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        specificity = []
        for i in range(len(cm)):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(np.delete(cm, i, axis=0)[:, i])
            specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        specificity = np.nanmean(specificity)
    
    return accuracy, precision, recall, specificity, f1, balanced_acc

#%% Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.show()

