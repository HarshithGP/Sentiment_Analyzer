
# Natural Language Processing

# Importing the libraries
import numpy as np
import pandas as pd
import csv

# Importing the dataset
dataset = pd.read_csv('train.csv', quoting = 3)

"""
with open("train.csv", 'r') as file:
    reviews = list(csv.reader(file))
"""
    
# Cleaning the texts
# Natural Language Processing  
# Apple Product Review Sentiment Analysis Using Naive Bayes Classifier

# Clean the text
# steps to clean the text
# 1. Replace non alphabetic charcaters with a space
# 2. Covert all characters to lowercase
# 3. Split the review into words
# 4. Remove common stop words - a, an, the, etc
# 5. Stem the words using Porter Stemmer to reduce sparsity
# 6. Join stemmed words to form a meaningful phrase
# 7. Build a corpus of a list of words from all reviews

import re #regular expressions
import nltk # natural language toolkit
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []
for i in range(0,479):
    review = re.sub('[^a-zA-Z]',' ',dataset['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray() #Sparse Matrix - independent variables

pos = 0
neg = 0
y = np.empty((0))
for j in range(0,479):
    if dataset.iloc[j,0] == 'Pos':
        pos = pos + 1
        y = np.append(y, 1)
    elif dataset.iloc[j,0] == 'Neg':
        neg = neg + 1
        y = np.append(y, 0)

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

#accuracy = classifier.score(X_test, y_test)
#print(round(accuracy*100,2),"%")

""" using Decision Tree Classifier """
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

""" using K Nearest Neighbours Classifier """
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(metric='minkowski',p=2)
classifier.fit(X_train, y_train)