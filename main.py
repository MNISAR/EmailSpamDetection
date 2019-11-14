import os
import re
import nltk
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB


dataset = pd.read_csv('emails.csv')
columns = dataset.columns
shape = dataset.shape

dataset.drop_duplicates(inplace = True)
dataset.shape
print (pd.DataFrame(dataset.isnull().sum()))


dataset['text']=dataset['text'].map(lambda text: text[1:])
dataset['text'] = dataset['text'].map(lambda text:re.sub('[^a-zA-Z0-9]+', ' ',text)).apply(lambda x: (x.lower()).split())
ps = PorterStemmer()
corpus=dataset['text'].apply(lambda text_list:' '.join(list(map(lambda word:ps.stem(word),(list(filter(lambda text:text not in set(stopwords.words('english')),text_list)))))))

cv = CountVectorizer()
X = cv.fit_transform(corpus.values).toarray()
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
classifier.fit(X_train , y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: \n",cm)

accuracy_score(y_test, y_pred) 
accuracy_score(y_test, y_pred,normalize=False) 


accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Mean: ",accuracies.mean())
print("Standard Deviation: ",accuracies.std())