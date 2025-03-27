import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained('./hate_speech_classifier')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit App UI
st.title("🚀 Hate Speech Classifier")
st.write("Enter a sentence to classify it as **Normal**, **Offensive**, or **Hate Speech**.")

# Input text
user_input = st.text_area("Enter your sentence here:")
li=[]

# Predict Button

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence!")
    else:
        # Tokenize and predict
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits).item()

        # Map class index to label
        class_map = {0: "Normal", 1: "Offensive", 2: "Hate Speech"}
        output=class_map.get(predicted_class, 'Unknown')
        li.append(output)
        st.success(f"✅ Predicted Class: **{class_map.get(predicted_class, 'Unknown')}**")

