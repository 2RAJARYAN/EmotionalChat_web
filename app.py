# app.py
import os
import torch
import streamlit as st
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
from huggingface_hub import InferenceClient


#  CONFIG 
HF_TOKEN = st.secrets["HF_TOKEN"]
EMOTION_MODEL_ID = st.secrets["EMOTION_MODEL_ID"]
LLM_MODEL_ID = st.secrets["LLM_MODEL_ID"]

EMOTION_THRESHOLD = 0.35
TOP_K = 3

# Login to HF
login(HF_TOKEN)


#  EMOTION MODEL 
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID,token = HF_TOKEN, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_MODEL_ID,
        device_map="cpu",
        token = HF_TOKEN
    )
    model.eval()
    return tokenizer, model


tokenizer, emo_model = load_emotion_model()


#  LABELS 
GOEMOTIONS_LABELS = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance",
    4: "approval", 5: "caring", 6: "confusion", 7: "curiosity",
    8: "desire", 9: "disappointment", 10: "disapproval", 11: "disgust",
    12: "embarrassment", 13: "excitement", 14: "fear", 15: "gratitude",
    16: "grief", 17: "joy", 18: "love", 19: "nervousness",
    20: "optimism", 21: "pride", 22: "realization", 23: "relief",
    24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
}


#  EMOTION DETECTION 
def detect_emotions(text: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cpu")

    with torch.no_grad():
        logits = emo_model(**inputs).logits.squeeze(0).cpu()
        probs = torch.sigmoid(logits).numpy()

    emotion_probs = [(GOEMOTIONS_LABELS[i], float(p)) for i, p in enumerate(probs)]
    emotion_probs.sort(key=lambda x: x[1], reverse=True)

    filtered = [(e, p) for e, p in emotion_probs if p >= EMOTION_THRESHOLD][:top_k]
    return filtered if filtered else [emotion_probs[0]]


#  HUGGINGFACE LLM CLIENT 
@st.cache_resource(show_spinner=False)
def load_llm_client():
    return InferenceClient(
        model=LLM_MODEL_ID,
        token=HF_TOKEN
    )


llm_client = load_llm_client()


def call_llm(prompt: str) -> str:
    """Use HF Inference Chat Completion API."""
    try:
        response = llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.6,
        )
        return response.choices[0].message["content"]

    except Exception as e:
        if "429" in str(e):
            return "HuggingFace API Rate Limit reached. Please wait."
        return f"[LLM Error] {str(e)}"


#  STREAMLIT UI 
st.set_page_config(page_title="Emotion-Aware Chatbot")
st.title("Emotion-Aware Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []


# Display chat history
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["text"])


#  USER INPUT 
user_text = st.chat_input("Type a message...")

if user_text:

    # Add user message
    st.session_state.history.append({"role": "user", "text": user_text})
    st.chat_message("user").write(user_text)

    # Emotion detection
    detected = detect_emotions(user_text)
    emo_str = ", ".join([f"{e} ({p:.2f})" for e, p in detected])

    st.markdown("### Detected Emotions")
    st.markdown(f"**{emo_str}**")

    # Conversation history for LLM
    conv_text = "\n".join([f"{m['role'].title()}: {m['text']}" for m in st.session_state.history])

    # Prompt for LLM
    prompt = f"""
                You are an empathetic AI assistant. If emotion tags are provided, adjust your tone accordingly.

                Detected emotions: {emo_str}

                Conversation so far:
                {conv_text}

                Now generate a helpful, empathetic response to the user's last message.
            """

    with st.spinner("Thinking..."):
        assistant_reply = call_llm(prompt)

    # Display and store reply
    st.session_state.history.append({"role": "assistant", "text": assistant_reply})
    st.chat_message("assistant").write(assistant_reply)
