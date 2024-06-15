from data_preparation import load_and_prepare_data
from text_processing import preprocess_text, download_nltk_resources
from model import SentimentModel, PrepareDataset_LSTM_GRU
from train import train_and_evaluate
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    # You might need to adjust paths and parameters
    data = load_and_prepare_data('/content/Womens Clothing E-Commerce Reviews.csv')
    data['processed_text'] = data['review_text'].apply(preprocess_text)
    download_nltk_resources()

    # Additional setup for tokenizer, etc.
    # Setup model, dataset, and train
    model = SentimentModel(vocab_size=5000, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=2, bidirectional=True, dropout=0.5).to(device)
    # Setup data loaders, train and evaluate the model
    train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device)

if __name__ == "__main__":
    main()
