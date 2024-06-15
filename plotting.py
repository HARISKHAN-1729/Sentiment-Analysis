import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def plot_confusion_matrices(cms, classes, titles, cmaps):
    fig, axes = plt.subplots(nrows=1, ncols=len(cms), figsize=(15, 5))
    for idx, cm in enumerate(cms):
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=cmaps[idx])
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title=titles[idx],
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

def plot_roc_curves(fpr_bert, tpr_bert, fpr_lstm, tpr_lstm):
    roc_auc_bert = auc(fpr_bert, tpr_bert)
    roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax1.plot(fpr_bert, tpr_bert, color='blue', lw=2, label=f'ROC curve BERT (area = {roc_auc_bert:.2f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve for BERT')
    ax1.legend(loc="lower right")
    ax2.plot(fpr_lstm, tpr_lstm, color='red', lw=2, label=f'ROC curve LSTM (area = {roc_auc_lstm:.2f})')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve for LSTM')
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

def plot_precision_recall_vs_threshold_side_by_side(precisions_bert, recalls_bert, thresholds_bert, precisions_lstm, recalls_lstm, thresholds_lstm):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax1.plot(thresholds_bert, precisions_bert[:-1], "b--", label="Precision", color="blue")
    ax1.plot(thresholds_bert, recalls_bert[:-1], "g-", label="Recall", color="green")
    ax1.set_title('BERT Precision-Recall vs Threshold')
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Value")
    ax1.legend(loc="upper left")
    ax1.set_ylim([0, 1])
    ax2.plot(thresholds_lstm, precisions_lstm[:-1], "b--", label="Precision", color="red")
    ax2.plot(thresholds_lstm, recalls_lstm[:-1], "g-", label="Recall", color="purple")
    ax2.set_title('LSTM Precision-Recall vs Threshold')
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Value")
    ax2.legend(loc="upper left")
    ax2.set_ylim([0, 1])
    plt.tight_layout()
    plt.show()
