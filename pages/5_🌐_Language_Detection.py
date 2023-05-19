import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Language Detection", page_icon="🌐")
st.subheader('🌐 Language Detection')

data = pd.read_csv('./pages/language_dataset_2.csv', encoding= 'unicode_escape')
print(data.head())

x = np.array(data["Text"])
y = np.array(data["Language"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=42)
model = MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)

user = st.text_input("Enter something: ")
data = cv.transform([user]).toarray()
button = st.button('Detect')
if button:
    output = model.predict(data)
    st.markdown(f"This language is: {output}")
    print(output)