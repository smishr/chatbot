# Importing the necessary libraries
import pandas as pd
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.classify import NaiveBayesClassifier

# Preprocessing the data
with open('product_data.txt', 'r') as file:
    text = file.read()

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]
    return dict([(token, True) for token in stemmed_tokens])

data = []
for sentence in nltk.sent_tokenize(text):
    data.append((preprocess(sentence), 'product'))

# Training the model
split = int(0.8 * len(data))
training_data = data[:split]
testing_data = data[split:]
classifier = NaiveBayesClassifier.train(training_data)

# Building the chatbot
def chatbot():
    print("How can I assist you?")
    print("Type 'exit' to quit")
    while True:
        query = input("You: ")
        if query == 'exit':
            break
        else:
            response = classifier.classify(preprocess(query))
            print("LLM: The product you may be interested in is: " + response)

# Testing the chatbot
chatbot()
