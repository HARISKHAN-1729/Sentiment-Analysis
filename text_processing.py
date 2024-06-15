import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def clean_text(text):
    """Convert text to lowercase, remove non-alphabetical characters, and trim spaces."""
    text = text.lower()
    text = ''.join(char for char in text if char.isalpha() or char.isspace())
    text = ' '.join(text.split())
    return text

def tokenize_text(text):
    """Split text into words or tokens."""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Eliminate common words that do not add significant meaning."""
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def stem_and_lemmatize(tokens):
    """Apply stemming and lemmatization."""
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stemmed = [stemmer.stem(token) for token in tokens]
    lemmatized = [lemmatizer.lemmatize(token) for token in stemmed]
    return lemmatized

def preprocess_text(text):
    """Complete text preprocessing workflow."""
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    tokens = remove_stopwords(tokens)
    normalized_tokens = stem_and_lemmatize(tokens)
    return ' '.join(normalized_tokens)
