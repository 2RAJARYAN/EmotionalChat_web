# app.py
import os
import requests
import streamlit as st
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_ollama import ChatOllama


# Configuration
OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_MODEL = "mistral"
EMOTION_MODEL_ID = "./final_model"
EMOTION_THRESHOLD = 0.35
TOP_K = 3


# Load Emotion Model Once
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_ID)
    model.eval()
    return tokenizer, model



tokenizer, emo_model = load_emotion_model()

# GoEmotions label mapping (clean)
GOEMOTIONS_LABELS = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral"
}


# Emotion Detection 

def detect_emotions(text: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits = emo_model(**inputs).logits.squeeze(0)
        probs = torch.sigmoid(logits).numpy()

    emotion_probs = [
        (GOEMOTIONS_LABELS[i], float(p))
        for i, p in enumerate(probs)
    ]

    # Sort and filter
    emotion_probs.sort(key=lambda x: x[1], reverse=True)
    filtered = [(e, p) for e, p in emotion_probs if p >= EMOTION_THRESHOLD][:top_k]

    return filtered if filtered else [emotion_probs[0]]



# Simple Ollama Call
def call_ollama(prompt: str) -> str:
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE, temperature=0.6)
    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)



# Streamlit UI
st.set_page_config(page_title="Emotion-aware Chatbot", page_icon="🤗")
st.title("Emotion-Aware Chatbot 💬🤗")

if "history" not in st.session_state:
    st.session_state.history = []

# Show history
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["text"])

# User input
user_text = st.chat_input("Type a message...")

if user_text:
    # Add user message
    st.session_state.history.append({"role": "user", "text": user_text})
    st.chat_message("user").write(user_text)

    # Detect emotion
    detected = detect_emotions(user_text)
    emo_str = ", ".join([f"{e} ({p:.2f})" for e, p in detected])
    st.markdown(f"**Detected emotions:** {emo_str} ")

    # Build conversation history string
    conv = "\n".join([f"{m['role'].title()}: {m['text']}" for m in st.session_state.history])

    # Prompt to LLM
    prompt = f"""
You are an empathetic assistant. Use detected emotions to respond kindly.

Detected emotions: {emo_str}

Conversation:
{conv}

Now reply to the user's last message in an empathetic and helpful tone:
"""

    with st.spinner("Thinking..."):
        assistant_text = call_ollama(prompt)

    # Save & display
    st.session_state.history.append({"role": "assistant", "text": assistant_text})
    st.chat_message("assistant").write(assistant_text)

