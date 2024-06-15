import torch
from model import SentimentModel, PrepareDataset_LSTM_GRU
from torch.utils.data import DataLoader

def binary_accuracy(predictions, labels):
    rounded_preds = torch.round(torch.sigmoid(predictions))
    correct = (rounded_preds == labels).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for texts, labels in iterator:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(texts).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for texts, labels in iterator:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=25):
    import os
    save_dir = '/content/drive/MyDrive/model'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'Lstm_gru_model.pth')
    best_val_acc = -float('inf')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        valid_loss, valid_acc = evaluate(model, test_loader, criterion, device)
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            torch.save(model.state_dict(), model_path)
            print(f'Model saved to {model_path}')
