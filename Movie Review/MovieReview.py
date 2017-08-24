import numpy as np
import pandas as pd
import glob
import re
import nltk # natural language toolkit
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

path='C:/Users/Harshith Guru Prasad/Desktop/Sentiment Analysis/aclImdb/train/Pos3000/*.txt'
files_pos=glob.glob(path)

corpus = []
for file in files_pos:
    with open(file, encoding="utf8") as f:
        review = f.read()
        review = re.sub('[^a-zA-Z]',' ',review).lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
    corpus.append(review)

path='C:/Users/Harshith Guru Prasad/Desktop/Sentiment Analysis/aclImdb/train/Neg3000/*.txt'
files_neg=glob.glob(path)

for file in files_neg:
    with open(file, encoding="utf8") as f:
        review = f.read()
        review = re.sub('[^a-zA-Z]',' ',review).lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray() #Sparse Matrix - independent variables

y = np.empty((0))
for i in range(3000):
    y = np.append(y, 1)
for i in range(3000):
    y = np.append(y, 0)

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
print("Accuracy = ", round(Accuracy,2), "%")

Precision = TP*100/(TP+FP)
print("Precision = ", round(Precision,2))

Recall = TP*100/(TP+FN)
print("Recall = ", round(Recall,2))

F1_Score = 2 * Precision * Recall / (Precision + Recall)
print("F1 Score = ",round(F1_Score),2)