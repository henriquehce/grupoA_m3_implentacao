"""
Inferencia do chatbot — suporta os dois modelos:
  - 'bow'        : modelo fiel ao artigo (TensorFlow + bag-of-words)
  - 'embeddings' : modelo moderno (sentence-transformers + similaridade)

Em ambos ha um limiar de confianca: abaixo dele, responde com o intent 'fallback'.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import pickle
import random
from pathlib import Path

import numpy as np

from preprocess import bag_of_words, load_intents

ROOT = Path(__file__).resolve().parents[1]
DATA_TOPICOS = ROOT / "data" / "intents_univali.json"   # modelo fiel (BoW)
DATA_FAQ = ROOT / "data" / "faq_univali.json"           # modelo moderno (FAQ real)
MODELS = ROOT / "models"


class ChatBot:
    def __init__(self, mode: str = "embeddings", threshold: float | None = None):
        """
        mode: 'bow' (artigo) ou 'embeddings' (moderno)
        threshold: confianca minima; abaixo disso -> fallback
        """
        self.mode = mode
        data_path = DATA_TOPICOS if mode == "bow" else DATA_FAQ
        self.intents = load_intents(data_path)
        self.responses = {it["tag"]: it["responses"] for it in self.intents["intents"]}
        self.threshold = threshold if threshold is not None else (0.6 if mode == "bow" else 0.40)

        if mode == "bow":
            self._load_bow()
        elif mode == "embeddings":
            self._load_embeddings()
        else:
            raise ValueError("mode deve ser 'bow' ou 'embeddings'")

    # --- carregamento ---------------------------------------------------------
    def _load_bow(self):
        from tensorflow.keras.models import load_model

        self.model = load_model(MODELS / "model_bow.keras")
        with open(MODELS / "bow_artifacts.pkl", "rb") as f:
            art = pickle.load(f)
        self.words = art["words"]
        self.classes = art["classes"]

    def _load_embeddings(self):
        from sentence_transformers import SentenceTransformer

        with open(MODELS / "embeddings_index.pkl", "rb") as f:
            idx = pickle.load(f)
        self.encoder = SentenceTransformer(idx["model_name"])
        self.pat_embeddings = idx["embeddings"]
        self.pat_labels = idx["labels"]

    # --- predicao da intencao -------------------------------------------------
    def predict_intent(self, text: str) -> tuple[str, float]:
        if self.mode == "bow":
            vec = np.array([bag_of_words(text, self.words)], dtype="float32")
            probs = self.model.predict(vec, verbose=0)[0]
            i = int(np.argmax(probs))
            return self.classes[i], float(probs[i])
        else:
            q = self.encoder.encode([text], normalize_embeddings=True)[0]
            sims = self.pat_embeddings @ q  # cosseno (vetores ja normalizados)
            i = int(np.argmax(sims))
            return self.pat_labels[i], float(sims[i])

    # --- resposta final -------------------------------------------------------
    def answer(self, text: str) -> dict:
        tag, conf = self.predict_intent(text)
        if conf < self.threshold:
            tag = "fallback"
        resp = random.choice(self.responses.get(tag, self.responses["fallback"]))
        return {"intent": tag, "confianca": round(conf, 3), "resposta": resp}


def _repl(mode: str):
    bot = ChatBot(mode=mode)
    print(f"Chatbot UNIVALI [{mode}] — digite 'sair' para encerrar.\n")
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
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "embeddings"
    _repl(mode)
