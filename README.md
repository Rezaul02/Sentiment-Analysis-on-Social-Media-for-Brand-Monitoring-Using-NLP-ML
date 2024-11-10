# Sentiment-Analysis-on-Social-Media-for-Brand-Monitoring-Using-NLP-ML


## ✅ Dataset link : https://www.kaggle.com/datasets/kazanova/sentiment140
![sentament](https://github.com/user-attachments/assets/120b142b-936d-4906-b10f-3e62a2a7036b)


This project performs Sentiment Analysis on Social Media Data using advanced machine learning models, specifically BERT. It aims to classify text into positive, neutral, and negative sentiments to help monitor brand perception on social platforms.

## ✅ Table of Contents
### Overview
Requirements
Installation
Usage
Project Structure
Model Prediction
Results
License
##  Overview
This project uses a BERT-based model to classify social media text into three sentiment categories: positive, neutral, and negative. The model is fine-tuned for sentiment classification tasks and can be applied to monitor brand sentiment and customer satisfaction based on social media posts.

### Requirements
Python 3.7 or later
TensorFlow 2.x
Transformers (Hugging Face library)
Pandas
NumPy
scikit-learn

## ✅ Installation
pip install -r requirements.txt
pip install tensorflow-hub tensorflow-text
pip install keras 
pip install tensorflow 

## ✅ Usage
Prepare the Dataset:

Ensure your dataset is in CSV format, with at least two columns: text (social media post text) and label (sentiment label, e.g., positive, neutral, or negative).
Map sentiment labels to numerical values (e.g., positive = 2, neutral = 1, negative = 0).

## ✅ Project Structure 
sentiment-analysis-brand-monitoring/
├── data/
│   └── Sintement Prediction.csv      
├── sentiment_analysis.py            
├── requirements.txt                 
├── README.md                      
└── models/
    └── saved_model/

##  ✅ Model Prediction
The model uses the predict_sentiment function to classify new text inputs. Here’s an overview of how this function works:
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)
    outputs = model(inputs)
    prediction = tf.argmax(outputs.logits, axis=1).numpy()[0]
    sentiment = [k for k, v in label_mapping.items() if v == prediction][0]
    return sentiment
### ✅ Explanation
Tokenization: The input text is tokenized using the BERT tokenizer.
Model Prediction: The tokenized text is passed to the model to get logits, which represent the probabilities for each class.
Class Extraction: The class with the highest probability is selected as the model's prediction.
Sentiment Mapping: The predicted class is converted to a sentiment label using a predefined label_mapping. 
## ✅ Results
The model should be able to classify the sentiment of social media posts with high accuracy after training. Accuracy metrics and other evaluation details are displayed at the end of model training.
![hfh](https://github.com/user-attachments/assets/4e7a3260-9b44-48ca-81f3-0356012f7406)




