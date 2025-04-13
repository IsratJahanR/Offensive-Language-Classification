# Toxicity Classification Project

## Overview

The **Toxicity Classification Project** aims to develop a machine learning model capable of detecting various types of toxic content in online feedback. The dataset consists of labeled comments with multiple categories of toxicity such as abusive, vulgar, menace, offense, and bigotry. The goal is to predict whether a given comment is **toxic** or not, considering both granular labels and the overarching toxicity classification.

## Dataset

The dataset consists of two main files:

- **train.csv** (Labeled Training Data)
  - **id**: Unique identifier for each comment
  - **feedback_text**: The feedback to be classified
  - **toxic**: 1 if the comment is toxic
  - **abusive**: 1 if the comment contains severe toxicity
  - **vulgar**: 1 if the comment contains obscene language
  - **menace**: 1 if the comment contains threats
  - **offense**: 1 if the comment contains insults
  - **bigotry**: 1 if the comment contains identity-based hate

- **test.csv** (Unlabeled data for prediction)

### Important Notes:
- Each label is binary (0 = offensive content not present, 1 = offensive content present), and multiple labels can be active for a single comment.
- The goal is to predict the **binary "toxic"** label for test data. While the training data includes more granular labels, these are used to help build the model and are not the final evaluation targets.

## Project Goal

The primary objective is to predict the presence of **toxic** content in feedback comments. The task is a multi-label classification problem where each comment can exhibit multiple types of toxicity. This project aims to implement a model that performs this classification and optimizes it for accuracy, recall, precision, and F1-score.

## Implementation Steps

### Step 1: Exploratory Data Analysis (EDA)
- Visualize label distribution across toxicity types
- Analyze sentence structure (length, word distribution, common words)
- Check for missing values or outliers

### Step 2: Text Preprocessing
- Tokenization: Split sentences into words
- Lowercasing: Convert text to lowercase
- Remove stop words, special characters, and punctuation
- Stemming/Lemmatization: Normalize words to their root form
- Feature Extraction: Convert text into numeric representations using TF-IDF, Word2Vec, or Transformer embeddings

### Step 3: Model Creation
- **Baseline Model**: Logistic Regression or Random Forest
- **Advanced Models**: LSTM or GRU for capturing sequential nature of text
- **Transformer-Based Models**: Fine-tune BERT or XLM for the task

### Step 4: Model Evaluation
- Evaluate using metrics such as accuracy, precision, recall, and F1-score
- Visualize confusion matrix and generate AUC-ROC curves to assess classification performance

### Step 5: Model Tuning and Optimization
- Experiment with different optimizers (Adam, SGD)
- Adjust learning rate, batch size, and number of epochs
- Use Grid Search or Random Search for hyperparameter tuning

## Evaluation Criteria

- **EDA Quality**: Depth of understanding of the dataset
- **Text Preprocessing**: Effectiveness of the cleaning process
- **Model Choice**: Suitability of the architecture
- **Performance Metrics**: Accuracy, precision, recall, F1-score, and AUC-ROC
- **Code Quality**: Clarity, modularity, and efficiency
- **Justification**: Clear explanation of the chosen modeling techniques

## Repository Structure

Offensive-Language-Classification/
│
├── task/
│   ├── model1_implementation.ipynb          # Baseline and Advanced Models (Logistic Regression, Random Forest, LSTM)
│   └── model2_implementation.ipynb          # Transformer Models (BERT, XLM)
│
├── Performance_Analysis_Report.pdf          # Analysis of model performance (e.g., confusion matrix, AUC-ROC, etc.)
│
├── Datasets         
│
├── README.md                               # Project overview and instructions
│
└── LICENSE                                 # License file (MIT or any other license you choose)

Clone the Repository

```bash
git clone https://github.com/IsratJahanR/Offensive-Language-Classification.git
cd Offensive-Language-Classification

---```



Developed by **Israt Jahan Reshma**  
Email: [israt.gstu@gmail.com](mailto:israt.gstu@gmail.com)  
GitHub: [IsratJahanR](https://github.com/IsratJahanR)
