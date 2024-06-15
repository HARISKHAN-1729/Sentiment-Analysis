# Sentiment-Analysis

This repository contains Python implementations of sentiment analysis models using BERT, LSTM, and GRU architectures to classify whether reviews recommend a product. The code is modular, optimized for clarity, extensibility, and practical application in real-world sentiment analysis tasks.

**Project Structure**
- data_preparation.py: Loads and prepares data for model training and evaluation.
- text_processing.py: Handles all text preprocessing requirements including tokenization and sequence padding.
- model.py: Defines both LSTM and GRU models along with the data handling specific to these architectures.
- train.py: Contains the training loops and evaluation logic for LSTM and GRU models.
- main.py: Main script to run the LSTM/GRU model workflows.
- data_setup_bert.py: Specialized data loading and preprocessing for BERT models.
- bert_model.py: Contains the BERT model definition and the dataset handling.
- train_bert.py: Scripts to train and evaluate the BERT model.
- bert_main.py: Main execution script for the BERT model operations.
- inference.py: Provides inference and evaluation functions for both BERT and LSTM/GRU models.
- plotting.py: Visualization functions for results such as ROC curves, confusion matrices, etc.
- evaluation.py: Uses inference and plotting to comprehensively assess and visualize model performance.

**Installation**

To install and run the models, follow these steps:

**1. Clone the Repository:**

   ```bash
   git clone https://github.com/HARISKHAN-1729/Sentiment-Analysis.git

   ```

**2. Set Up a Python Environment:
(Recommended: Use Python 3.8 or newer)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```

**3. Install Requirements:**
   
   ```bash
    pip install -r requirements.txt
```

**Usage**
For BERT Model:

```bash
python bert_main.py
```

For LSTM/GRU Model:

```bash
python main.py
```

To Evaluate Models and Generate Plots:

```bash
python evaluation.py
```

Make sure to set dataset paths and any necessary configurations properly in the scripts.

**Additional Notes**

- Modify file paths according to your local or cloud environment.
- Adjust model parameters and other configurations to suit your data and system capabilities.

  
**Contributing**
- Feel free to fork the repository, make improvements, and submit a pull request. We appreciate your contributions!

**License**
This project is licensed under the MIT License - see the LICENSE file for more details.
