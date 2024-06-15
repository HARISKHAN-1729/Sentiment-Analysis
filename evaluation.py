import torch
from inference import inference_and_evaluate
from plotting import plot_confusion_matrices, plot_roc_curves, plot_precision_recall_vs_threshold_side_by_side, plot_histograms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, precision_recall_curve, roc_curve

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_bert_path = '/content/drive/MyDrive/model/bert_model.pth'
    model_lstm_path = '/content/drive/MyDrive/model/lstm_model.pth'
    test_loader_bert = # Define or load your BERT test_loader
    test_loader_lstm = # Define or load your LSTM/GRU test_loader
    
    preds_bert, labels_bert, loss_bert, acc_bert = inference_and_evaluate(model_bert, model_bert_path, test_loader_bert, device, 'BERT')
    preds_lstm, labels_lstm, loss_lstm, acc_lstm = inference_and_evaluate(model_lstm, model_lstm_path, test_loader_lstm, device, 'LSTM/GRU')

    metrics_bert = precision_recall_fscore_support(labels_bert, preds_bert, average='binary')
    metrics_lstm = precision_recall_fscore_support(labels_lstm, preds_lstm, average='binary')

    precision_bert, recall_bert, thresholds_bert = precision_recall_curve(labels_bert, preds_bert)
    precision_lstm, recall_lstm, thresholds_lstm = precision_recall_curve(labels_lstm, preds_lstm)

    plot_precision_recall_vs_threshold_side_by_side(precision_bert, recall_bert, thresholds_bert, precision_lstm, recall_lstm, thresholds_lstm)

    cm_bert = confusion_matrix(labels_bert, preds_bert)
    cm_lstm = confusion_matrix(labels_lstm, preds_lstm)
    plot_confusion_matrices([cm_bert, cm_lstm], classes=['Not Recommended', 'Recommended'], titles=['Confusion Matrix BERT', 'Confusion Matrix LSTM'], cmaps=[plt.cm.Blues, plt.cm.Oranges])

    fpr_bert, tpr_bert, _ = roc_curve(labels_bert, preds_bert)
    fpr_lstm, tpr_lstm, _ = roc_curve(labels_lstm, preds_lstm)
    plot_roc_curves(fpr_bert, tpr_bert, fpr_lstm, tpr_lstm)

    bert_metrics = [loss_bert, acc_bert, metrics_bert[2], metrics_bert[0], metrics_bert[1]]
    lstm_metrics = [loss_lstm, acc_lstm, metrics_lstm[2], metrics_lstm[0], metrics_lstm[1]]
    metrics_data = [bert_metrics, lstm_metrics]
    metrics_names = ['Loss', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
    model_labels = ['BERT', 'LSTM/GRU']
    plot_histograms(metrics_data, model_labels, metrics_names)

    print("BERT Metrics: Precision:", metrics_bert[0], "Recall:", metrics_bert[1], "F1 Score:", metrics_bert[2], "Accuracy:", acc_bert)
    print("LSTM Metrics: Precision:", metrics_lstm[0], "Recall:", metrics_lstm[1], "F1 Score:", metrics_lstm[2], "Accuracy:", acc_lstm)

if __name__ == "__main__":
    main()
