import numpy as np
import pandas as pd
import itertools
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Read the data
df=pd.read_csv('D:\\utcc\\Ai\\project\\news.csv')
#Get shape and head
df.shape
df.head()

#DataFlair - Get the labels
labels=df.label
labels.head()

#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

# Save the model
joblib.dump(pac, 'fake_news_classifier_model.joblib')

# Load the model
pac_loaded = joblib.load('fake_news_classifier_model.joblib')

# Example prediction
new_data = ["Is google and YouTube in the Hillary's purse?","Jeb Bush Picks Up Endorsement From Lindsey Graham","Real Disclosure! Secret Alien Base Found In Moon's Tycho Crater"]
tfidf_new_data = tfidf_vectorizer.transform(new_data)
predictions = pac_loaded.predict(tfidf_new_data)
print(predictions)