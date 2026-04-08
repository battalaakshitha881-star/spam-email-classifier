import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Train the model
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Streamlit UI
st.title("🚨 Spam Email Classifier")
st.write("Enter a message below to check if it's spam!")

user_input = st.text_area("Your Message:")

if st.button("Check"):
    if user_input:
        vec = vectorizer.transform([user_input])
        result = model.predict(vec)[0]
        if result == 1:
            st.error("🚨 SPAM!")
        else:
            st.success("✅ Not Spam (Ham)")