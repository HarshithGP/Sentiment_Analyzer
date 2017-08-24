# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_pred)

TP = confusionMatrix[1][1]
FP = confusionMatrix[0][1]

TN = confusionMatrix[0][0]
FN = confusionMatrix[1][0]

Accuracy = (TP+TN)*100/(TP+TN+FP+FN)
print("Accuracy = ", round(Accuracy,2),"%")

Precision = TP*100/(TP+FP)
print("Precision = ", round(Precision,2),"%")

Recall = TP*100/(TP+FN)
print("Recall = ", round(Recall,2),"%")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy = round(accuracy * 100,2)
print("Accuracy = %0.2f"%accuracy,"%")