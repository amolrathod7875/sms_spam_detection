import streamlit as st
import pickle
import numpy as np
import time

# 1. Page Config (Must be the first command)
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📲",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Styling & Animations
# This injects custom HTML/CSS to style buttons, backgrounds, and add animations.
st.markdown("""
<style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);
    }
    
    /* Input Text Area styling */
    .stTextArea textarea {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #ced4da;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Predict Button Styling with Hover Animation */
    .stButton>button {
        background: linear-gradient(45deg, #4b6cb7, #182848);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.2);
        background: linear-gradient(45deg, #182848, #4b6cb7);
    }

    /* Result Card Styling */
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Resources
@st.cache_resource
def load_resources():
    try:
        # Ensure these files exist in your directory
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer
    except Exception as e:
        return None, None

model, vectorizer = load_resources()

# 4. Sidebar content
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913963.png", width=80)
    st.title("Spam Detector")
    st.markdown("---")
    st.write("This AI model detects spam messages with **98% Accuracy**.")
    st.write("### How it works:")
    st.caption("1. Enter your SMS text.")
    st.caption("2. The model analyzes keywords.")
    st.caption("3. It predicts if it's Spam or Ham.")
    st.markdown("---")
    st.write("Created by **Amol Rathod**")

# 5. Main App UI
st.title("📲 SMS Spam Classification")
st.markdown("### Protect your inbox from unwanted messages")
st.write("Paste your message below to analyze its content.")

# Layout: Input section
col1, col2 = st.columns([4, 1]) 

sms_text = st.text_area("Message Content", height=150, placeholder="e.g. You have won a lottery! Click here...")

if st.button("🔍 Analyze Message"):
    if not sms_text:
        st.warning("⚠️ Please enter some text to analyze.")
    elif model is None or vectorizer is None:
        st.error("❌ Model files not found. Please check your directory.")
    else:
        # Progress bar animation for "thinking" effect
        progress_text = "Analyzing text patterns..."
        my_bar = st.progress(0, text=progress_text)
        
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        time.sleep(0.5) # Short pause for effect
        my_bar.empty() # Remove progress bar

        # Prediction Logic
        vector_input = vectorizer.transform([sms_text])
        prediction = model.predict(vector_input)[0]
        prediction_proba = model.predict_proba(vector_input)[0] # Get probabilities
        
        # Display Results with Custom HTML/CSS Cards
        if prediction == 1:
            confidence = round(prediction_proba[1] * 100, 2)
            st.markdown(f"""
            <div class="result-card" style="background-color: #ffebee; border: 2px solid #ffcdd2; color: #b71c1c;">
                <h1>🚨 SPAM DETECTED</h1>
                <h3>Confidence Score: {confidence}%</h3>
                <p>This message contains patterns typically found in spam.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence = round(prediction_proba[0] * 100, 2)
            st.markdown(f"""
            <div class="result-card" style="background-color: #e8f5e9; border: 2px solid #c8e6c9; color: #1b5e20;">
                <h1>✅ HAM (Safe Message)</h1>
                <h3>Confidence Score: {confidence}%</h3>
                <p>This message looks like a normal conversation.</p>
            </div>
            """, unsafe_allow_html=True)