import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os


print("Ensuring NLTK assets are downloaded...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

if not os.path.exists('trained_model'):
    os.makedirs('trained_model')


def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- Model Training ---
print("Starting model training...")
df = pd.read_csv('data.csv')
print(f"Loaded {len(df)} records from data.csv.")

df.dropna(subset=['text'], inplace=True) # Ensure no empty text rows
df['processed_text'] = df['text'].apply(preprocess_text)

X = df['processed_text']
y = df['category']

vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(X)
print("TF-IDF features created.")


X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples.")

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Model trained.")


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on test set: {accuracy * 100:.2f}%")


with open('trained_model/logistic_regression.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('trained_model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully to 'trained_model/'")

if accuracy == 0.0:
    print("Warning: Model accuracy is 0%. Consider adding more diverse data to 'data.csv'.")
else:
    print("\n✅ Training complete and model accuracy is above 0%!")