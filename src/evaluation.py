import os
import csv
import numpy as np
import matplotlib.pyplot as plt



def compute_and_save_metrics(results_csv, results_dir):
    """
    Reads results CSV, computes overall accuracy, class-wise precision and recall,
    writes metrics.csv and metrics.txt, and saves a confusion matrix heatmap.
    """
    # Loading predictions and ground truths
    y_true = []
    y_pred = []
    
    with open(results_csv, newline='') as f:
        reader = csv.DictReader(f)
    
        for row in reader:
            y_true.append(int(row['ground_truth']))
            y_pred.append(int(row['predicted']))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Identifying classes
    classes = sorted(set(y_true))
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Building confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for t, p in zip(y_true, y_pred):
    
        if p in class_to_idx and t in class_to_idx:
            cm[class_to_idx[t], class_to_idx[p]] += 1

    # Computing overall accuracy
    total = y_true.shape[0]
    correct = (y_true == y_pred).sum()
    overall_acc = correct / total if total > 0 else 0

    # Computing class-wise precision, recall, support
    precision = []
    recall = []
    support = []
    
    for idx, c in enumerate(classes):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        support.append(cm[idx, :].sum())

    # Saving metrics.csv
    metrics_csv = os.path.join(results_dir, 'metrics.csv')
    
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'precision', 'recall', 'support'])
    
        for c, p, r, s in zip(classes, precision, recall, support):
            writer.writerow([c, f'{p:.4f}', f'{r:.4f}', s])
    
        writer.writerow(['overall_accuracy', f'{overall_acc:.4f}', '', total])

    # Save metrics.txt
    metrics_txt = os.path.join(results_dir, 'metrics.txt')
    
    with open(metrics_txt, 'w') as f:
        f.write(f'Overall accuracy: {overall_acc:.4f}\n')
        f.write('Class-wise precision and recall:\n')
    
        for c, p, r in zip(classes, precision, recall):
            f.write(f'Class {c}: Precision={p:.4f}, Recall={r:.4f}\n')

    # Plot and save confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Annotate counts
    thresh = cm.max() / 2
    
    for i in range(n_classes):
    
        for j in range(n_classes):
            plt.text(j, i, cm[i, j],
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()

    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

    print(f"\n[Done] Metrics saved to {metrics_csv} and {metrics_txt}")
    print(f"\n[Done] Confusion matrix image saved to {cm_path}")