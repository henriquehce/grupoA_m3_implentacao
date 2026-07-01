"""
Inferencia do chatbot — modelo fiel ao artigo (Salloum et al., 2024):
TensorFlow + bag-of-words.

Ha um limiar de confianca: abaixo dele, responde com o intent 'fallback'.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pickle
import random
from pathlib import Path

import numpy as np

from preprocess import bag_of_words, load_intents

ROOT = Path(__file__).resolve().parents[1]
DATA_TOPICOS = ROOT / "data" / "intents_univali.json"   # modelo fiel (BoW)
MODELS = ROOT / "models"


class ChatBot:
    def __init__(self, threshold: float = 0.6):
        """threshold: confianca minima; abaixo disso -> fallback."""
        self.intents = load_intents(DATA_TOPICOS)
        self.responses = {it["tag"]: it["responses"] for it in self.intents["intents"]}
        self.threshold = threshold

        from tensorflow.keras.models import load_model

        self.model = load_model(MODELS / "model_bow.keras")
        with open(MODELS / "bow_artifacts.pkl", "rb") as f:
            art = pickle.load(f)
        self.words = art["words"]
        self.classes = art["classes"]

    # --- predicao da intencao -------------------------------------------------
    def predict_intent(self, text: str) -> tuple[str, float]:
        vec = np.array([bag_of_words(text, self.words)], dtype="float32")
        probs = self.model.predict(vec, verbose=0)[0]
        i = int(np.argmax(probs))
        return self.classes[i], float(probs[i])

    # --- resposta final -------------------------------------------------------
    def answer(self, text: str) -> dict:
        tag, conf = self.predict_intent(text)
        if conf < self.threshold:
            tag = "fallback"
        resp = random.choice(self.responses.get(tag, self.responses["fallback"]))
        return {"intent": tag, "confianca": round(conf, 3), "resposta": resp}


def _repl():
    bot = ChatBot()
    print("Chatbot UNIVALI [BoW] — digite 'sair' para encerrar.\n")
    while True:
        try:
            text = input("Voce: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"sair", "exit", "quit"}:
            break
        out = bot.answer(text)
        print(f"Bot ({out['intent']} | {out['confianca']}): {out['resposta']}\n")


if __name__ == "__main__":
    _repl()
