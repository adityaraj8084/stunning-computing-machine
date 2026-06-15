import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("tweets.csv")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["cleaned_tweet"] = df["tweet"].apply(clean_text)

X = df["cleaned_tweet"]
y = df["sentiment"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

while True:
    tweet = input("Enter Tweet (or quit): ")

    if tweet.lower() == "quit":
        break

    cleaned = clean_text(tweet)
    transformed = vectorizer.transform([cleaned])

    prediction = model.predict(transformed)

    print("Sentiment:", prediction[0])
