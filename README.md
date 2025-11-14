# Head: A Modular Emotion-Aware Middleware for LLM-Powered Chatbots

## 🌟 Overview

Head is a modular middleware architecture designed to make Large Language Models (LLMs) **emotionally aware**. Instead of retraining or fine‑tuning large models, Head acts as a plug‑and‑play component that analyzes user input, extracts emotional signals, and passes emotion‑tagged metadata to any downstream chatbot model.

This repository contains:

* 🧠 **Head Module** — A BERT-based multi‑label emotion classifier trained on GoEmotions.
* 💬 **Emotion-Aware Chat Application** — A Streamlit UI that integrates the Head module with a locally running LLM using Ollama.
* ⚙️ **Middleware Architecture** — Clean, modular design for easy integration into any chatbot system.

Our goal is to create more *empathetic*, *context‑aware*, and *human‑aligned* conversational agents.

---

## 🚀 Features

* **Multi-label Emotion Classification**
  Powered by a fine‑tuned BERT model on GoEmotions (simplified variant, 28 emotions).

* **Probabilistic Emotion Outputs**
  Each input message returns emotion scores with threshold-based multi-label selection.

* **Plug-and-Play Middleware**
  Head can sit between *any* input and chatbot model.

* **Interactive Streamlit UI**
  Complete frontend for testing and demonstrating emotion-aware interactions.

---

## 🧩 Architecture

```
User → Head (Emotion Extractor) → Emotion Tags → LLM (Ollama) → Chatbot Response
```

### 🔹 Head Module

* Built on `AutoModelForSequenceClassification` from HuggingFace.
* Multi-label BCEWithLogits setup.
* Thresholding per emotion class.

### 🔹 LLM Response Generator

* Uses Hugging Face Chat Completion APIs.
* Injects emotion tags + user text into the prompt.

### 🔹 Why Middleware?

* No need to fine‑tune LLMs.
* Works with *any* model.
* Modular, interpretable, and expandable (audio/video/physiological modalities later).

---

## 🖥️ Streamlit Demo

The app provides:

* Input text box
* Predicted emotions + probabilities
* Emotion-aware LLM response
* Debug panel for prompt inspection

Run with:

```bash
streamlit run app.py
```
---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/2RAJARYAN/EmotionalChat_web.git
cd EmotionalChat_web
```
### 2. Create a vene

```bash
python -m venv .venv
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup .streamlit/secrets.toml
- Create a Folder name .streamlit
- A File called secrets.toml
- This store the HF_TOKEN, EMOTION_MODEL_ID, LLM_MODEL_ID

---

## 🧠 Future Scope

* Multimodal Head (text + audio + video)
* Novel emotion decoding algorithms
* Benchmark suite for emotional coherence and empathy
* Publication in NLP/affective computing venues

---

## 📄 License

MIT License.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## ⭐ Acknowledgements

* Google’s GoEmotions dataset
* HuggingFace Transformers

---

If you like this project, consider giving us a ⭐ on GitHub!
