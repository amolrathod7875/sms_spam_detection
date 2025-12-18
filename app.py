import streamlit as st
import pickle
import numpy as np

# 1. Load the trained model and vectorizer
# We use @st.cache_resource so it loads only once, not every time the page refreshes
@st.cache_resource
def load_resources():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {e}")
        return None, None

model, vectorizer = load_resources()

# 2. App Title and Description
st.title("📱 SMS Spam Predictor")
st.write("Enter a message below to check if it looks like **Spam** or **Ham** (Normal).")

# 3. Input Text Area
sms_text = st.text_area("Enter SMS Text", height=150)

# 4. Predict Button
if st.button("Predict"):
    if not sms_text:
        st.warning("Please enter some text to analyze.")
    elif model is None or vectorizer is None:
        st.error("Model files not found. Please check your .pkl files.")
    else:
        # Preprocess: Vectorize the input
        vector_input = vectorizer.transform([sms_text])
        
        # Predict
        prediction = model.predict(vector_input)[0]
        
        # Display Result
        st.subheader("Result:")
        if prediction == 1:
            st.error("🚨 SPAM")
        else:
            st.success("✅ HAM (Not Spam)")