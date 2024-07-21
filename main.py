import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv("Fake_Real.csv")

df = df.dropna(subset=["text"])

x_train, x_test, y_train, y_test = train_test_split(
    df["text"], df["target"], test_size=0.2, random_state=7
)

tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)

tfidf_test = tfidf_vectorizer.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)

pac.fit(tfidf_train, y_train)

y_pred = pac.predict(tfidf_test)

score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score*100,2)}%")

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
