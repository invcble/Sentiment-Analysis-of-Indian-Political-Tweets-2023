# Sentiment Analysis of Indian Political Tweets using LSTM

This project was developed as part of the Advanced Machine Learning course requirement, focusing on analyzing the sentiment of Indian political tweets using a Long Short-Term Memory (LSTM) model. The study efficiently incorporates Tweepy for data collection, utilizes a labeled dataset from Kaggle, and applies Natural Language Processing (NLP) techniques along with Global Vector (GloVe) word embeddings for data preprocessing. The project's methodology offers a comprehensive approach to sentiment analysis, categorizing tweets into positive, neutral, or negative sentiments and achieving an impressive model accuracy of 96%.

## Dataset

The dataset for this project was sourced from Kaggle, specifically tailored for sentiment analysis of Indian political tweets. It plays a crucial role in training and validating our LSTM model to ensure accurate sentiment classification.

- **Dataset Source**: [Kaggle: Indian Political Tweets Sentiment Analysis](https://www.kaggle.com/datasets/your-dataset-link)

## Models and Accuracy

The LSTM model was employed for sentiment classification, with the following performance metrics:

- Precision
- Recall
- F1-Score

These metrics indicate the model's strong ability to classify the sentiment of tweets accurately. The performance is summarized as follows:

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.95      | 0.96   | 0.96     | 7276    |
| 1         | 0.95      | 0.97   | 0.96     | 6935    |
| 2         | 0.97      | 0.94   | 0.95     | 7089    |
|           |           |        |          |         |
| Accuracy  |           |        | 0.96     | 21300   |
| Macro Avg | 0.96      | 0.96   | 0.96     | 21300   |
| Weighted Avg | 0.96   | 0.96   | 0.96     | 21300   |

- [**Click here for Source Codes**](https://github.com/invcble/Sentiment-Analysis-of-Indian-Political-Tweets-2023/tree/ec49ca15b794566ff53c79ab2bfa2437bc95431b/Source%20codes)
- [**Click here for Project Report**](https://github.com/invcble/Sentiment-Analysis-of-Indian-Political-Tweets-2023/blob/ec49ca15b794566ff53c79ab2bfa2437bc95431b/Project_Report_7thSEM.pdf)

## Setup and Running the Project

To replicate and run this project, follow these steps:

1. **Collect Tweets**: Use Tweepy to collect Indian political tweets. A guide for setting up Tweepy and collecting tweets is provided in the source codes.
2. **Prepare the Dataset**: Download the labeled dataset from Kaggle and preprocess it using the provided scripts for GloVe embeddings.
3. **Train and Evaluate the LSTM Model**: Follow the instructions in the LSTM model implementation folder to train and evaluate the sentiment analysis model.

## Requirements

This project is designed to be run in a Python environment with support for Jupyter Notebooks or Google Colaboratory. Key dependencies include TensorFlow, Keras, Tweepy, Pandas, NumPy, and Matplotlib.

## Acknowledgments

We extend our gratitude to the academic staff and my peers at Bengal Institute of Technology for their invaluable feedback and support. Special thanks to the Kaggle community for providing a comprehensive dataset for sentiment analysis of Indian political tweets.

