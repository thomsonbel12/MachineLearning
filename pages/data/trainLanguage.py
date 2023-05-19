import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Step 1: Preprocess the data
data = pd.read_csv('pages\data\language.csv')
text_data = data['Text'].tolist()
language_data = data['Language'].tolist()

# Step 2: Tokenization and Formatting
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data)

# Step 3: Train the Language Model
model = MultinomialNB()
model.fit(X, language_data)

# Step 4: Save the Trained Model as a Pickle File
model_path = 'pages\data\language_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model, file)