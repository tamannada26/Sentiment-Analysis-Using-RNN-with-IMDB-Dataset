
# Sentiment-Analysis-Using-Recurrent-Neural-Networks-RNN-with-IMDB-Dataset



## Overview

This project demonstrates sentiment analysis using **Recurrent Neural Networks (RNN)** for classifying movie reviews from the **IMDB dataset** as either positive or negative. The dataset consists of 25,000 labeled movie reviews (positive and negative), which are used for training and testing.

The objective of this project is to build and evaluate different models for sentiment classification, including:
- A **Simple RNN**
- An **RNN after Hyperparameter Tuning**
- A **Feedforward Neural Network (FFN)**

## Dataset

The dataset used for this project is the **IMDB movie reviews dataset**, which contains:
- **25,000** labeled reviews for training
- **25,000** labeled reviews for testing

Each review is labeled with either a **0** (negative sentiment) or a **1** (positive sentiment).

## Project Structure

### 1. **Data Preprocessing**
   - **Text Tokenization**: The text data is tokenized into sequences of integers using the Keras `Tokenizer`.
   - **Padding Sequences**: Sequences are padded to a fixed length to ensure consistent input dimensions for the models.

### 2. **Model Architectures**
   - **Recurrent Neural Networks (RNNs)**: Initially, simple RNN models were trained for sentiment classification.
   - **Hyperparameter Tuning with Grid Search**: After evaluating the simple RNN model, hyperparameter tuning was done using Grid Search to optimize the model's performance.
   - **Feedforward Neural Network (FFN)**: After tuning the RNN, a simple feedforward neural network was trained as an additional model for comparison.

### 3. **Model Training**
   - The models are trained using the training dataset, with hyperparameters tuned to improve performance.
   - Binary cross-entropy loss and accuracy metrics are used to evaluate model performance.

### 4. **Model Evaluation**
   - The models are evaluated on a test set using metrics like **accuracy**, **precision**, **recall**, and **F1-score**.
   - Classification reports and confusion matrices are generated to assess each model's performance.

### 5. **Model Comparison**
   - We compare the results from the three models: Simple RNN, Hyperparameter-Tuned RNN, and Feedforward Neural Network.
   - Insights are drawn based on model accuracy, ability to handle long-term dependencies, and generalization to unseen data.

## Requirements

To run this project, you need to install the required dependencies. You can install them using the following:


### Required Libraries:
- TensorFlow/Keras
- Numpy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Future Work

- **Hyperparameter Tuning**: Further explore hyperparameter tuning to improve the models' performance.
- **Attention Mechanisms**: Integrate attention mechanisms to improve the LSTM model's ability to focus on important words in the review.
- **Transformer Models**: Explore transformer-based architectures such as BERT for improved performance on text classification tasks.


## How to Run

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/tamannada26/sentiment-analysis-rnn-imdb.git


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IMDB dataset for sentiment analysis
- Keras/TensorFlow for deep learning models
- Scikit-learn for evaluation metrics and tools



