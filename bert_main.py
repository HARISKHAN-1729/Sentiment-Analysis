from data_setup_bert import load_data, split_data
from bert_model import BertSentimentModel, Prepare_BertDataset
from train_bert import train_bert, evaluate_bert
import torch

def main():
    PRETRAINED_MODEL_NAME = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    max_len = 58
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    texts, labels = load_data('/content/processed_reviews.csv')
    X_train, X_test, y_train, y_test = split_data(texts, labels)
    
    train_data = Prepare_BertDataset(X_train, y_train, tokenizer, max_len)
    test_data = Prepare_BertDataset(X_test, y_test, tokenizer, max_len)
    
    train_loader_bert = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader_bert = DataLoader(test_data, batch_size=32)
    
    model_bert = BertSentimentModel(PRETRAINED_MODEL_NAME).to(device)
    optimizer_bert = AdamW(model_bert.parameters(), lr=0.00001, weight_decay=0.01)
    criterion = BCEWithLogitsLoss().to(device)
    
    # Train and evaluate
    for epoch in range(25):
        print(f'Epoch {epoch+1}/25')
        train_loss, train_acc = train_bert(model_bert, train_loader_bert, optimizer_bert, criterion, device)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        valid_loss, valid_acc = evaluate_bert(model_bert, test_loader_bert, criterion, device)
        print(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')

if __name__ == "__main__":
    main()
