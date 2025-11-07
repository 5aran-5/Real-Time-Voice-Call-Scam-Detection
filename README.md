# 🎙️ Real-Time Voice Call Scam Detection using Audio Stream Chunking and Hybrid NLP Models

> A real-time voice monitoring system that detects potential scam or fraudulent conversations using a hybrid AI pipeline combining **speech recognition**, **contextual NLP embeddings**, and **machine learning classification**.

---

## 🧩 Overview

This project simulates a **real-time voice call monitoring** system that flags potential scam conversations as they occur.  
It integrates **OpenAI Whisper** for speech-to-text transcription and **DistilBERT** for contextual understanding, followed by a **custom ML classifier** that predicts scam probability in near real-time.

The pipeline processes live-like audio streams using **multithreading**, ensuring low-latency inference while maintaining contextual awareness.

---

## ⚙️ Technical Workflow

### 🎧 Audio Input Simulation
- Loads pre-recorded `.wav` or `.mp3` files and resamples them to 16 kHz.
- Splits audio into **overlapping chunks** (e.g., 3s duration, 1.5s hop) to simulate continuous input.

### 🔄 Real-Time Processing Pipeline (Multithreaded)
| Thread | Task |
|:-------|:------|
| 🎵 **Audio Thread** | Streams chunks sequentially with real-time delays. |
| ✍️ **Transcription Thread** | Converts audio chunks to text using **Whisper**. |
| 🧠 **Classification Thread** | Combines recent transcripts and uses **DistilBERT embeddings** + ML classifier to predict scam likelihood. |

### 🧮 Classification Model
- **DistilBERT**: Generates contextual embeddings of recent transcript segments.
- **Scikit-learn Classifier**: Trained to predict scam probability using extracted embeddings.
- **Alert System**: Flags calls when probability exceeds a threshold (default `0.6`).

---

## 🚀 Key Features

- 🧵 **Multithreaded architecture** simulating real-time call streams.  
- 🤖 **Hybrid AI pipeline** combining Whisper (Speech) + DistilBERT (Context) + ML classifier (Decision).  
- 🧩 **Context-aware classification** using multi-chunk aggregation.  
- ⚡ **Configurable parameters**: chunk duration, hop size, alert threshold, Whisper model type.  
- 💻 **GPU-accelerated inference** via PyTorch + CUDA.  
- 🧠 Modular & extendable for fraud detection, moderation, or customer analytics.  

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|:----------|:------------------|
| **Language** | Python |
| **Speech Recognition** | [OpenAI Whisper](https://github.com/openai/whisper) |
| **NLP Embeddings** | [Hugging Face Transformers (DistilBERT)](https://huggingface.co/distilbert-base-uncased) |
| **ML Frameworks** | PyTorch, scikit-learn, joblib |
| **Audio Processing** | librosa, numpy |
| **Utilities** | threading, argparse |

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/5aran-5/Real-Time-Voice-Call-Scam-Detection.git
cd Real-Time-Voice-Call-Scam-Detection
