import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import os
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.val_loss_min = np.Inf
    def __call__(self, val_loss, model, save_path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"Early stopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, save_path):
        print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss
        
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = torch.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=1))
    
#########################################
# Plotting Functions
#########################################
def plot_radar_chart(metrics, epoch, save_path):
    categories = ['Accuracy', 'F1', 'Precision', 'Recall', 'AUC', 'AUPRC']
    values = [metrics['accuracy'], metrics['f1'], metrics['precision'], metrics['recall'], metrics['auc'], metrics['auprc']]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticklabels([])
    ax.set_title(f"Metrics Radar Plot - Epoch {epoch}")
    plt.savefig(os.path.join(save_path, f"radar_plot_epoch_{epoch}.png"))
    plt.close()

def compute_auprc_auroc(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
    auprc = auc(recall, precision)
    return fpr, tpr, precision, recall, auroc, auprc

def plot_auprc_auroc(fpr, tpr, precision, recall, auroc, auprc, save_path):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC: {auroc:.4f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, "auroc.png"))
    plt.close()
    plt.figure()
    plt.plot(recall, precision, label=f"AUPRC: {auprc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(save_path, "auprc.png"))
    plt.close()