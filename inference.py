import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, precision_recall_fscore_support

def inference_and_evaluate(model, model_path, test_loader, device, model_type):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()

    all_preds, all_labels = [], []
    epoch_loss, epoch_acc = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            if model_type == 'BERT':
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                outputs = model(input_ids, attention_mask).squeeze(1)
            else:  # LSTM/GRU
                texts, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(texts).squeeze(1)

            loss = criterion(outputs, labels.float())
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            epoch_loss += loss.item()
            correct = (preds == labels).float()
            acc = correct.sum().item() / len(correct)
            epoch_acc += acc
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    loss = epoch_loss / len(test_loader)
    accuracy = epoch_acc / len(test_loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return all_preds, all_labels, loss, accuracy
